import sqlite3
from fastapi import HTTPException
import os
import json
from my_tools import TOOL_CLASSES
from sqlalchemy import create_engine, text
from datetime import datetime
import logging
from typing import Optional
# from routes.db_routes import AgentPayload, TaskPayload, CrewPayload
logger = logging.getLogger(__name__)


# If you have an environment variable DB_URL for Postgres, use that. 
# Otherwise, fallback to local SQLite file: 'sqlite:///crewai.db'
DEFAULT_SQLITE_URL = 'sqlite:///crewai.db'
DB_URL = os.getenv('DB_URL', DEFAULT_SQLITE_URL)

# Create a SQLAlchemy Engine.
# For example, DB_URL could be:
#   "postgresql://username:password@hostname:5432/dbname"
# or fallback to: "sqlite:///crewai.db"
engine = create_engine(DB_URL, echo=False)

def get_db_connection():
    # conn = sqlite3.connect(DB_NAME)
    # conn.row_factory = sqlite3.Row
    # return conn
    """
    Return a context-managed connection from the SQLAlchemy engine.
    """
    return engine.connect()

def create_tables():
    create_sql = text('''
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            entity_type TEXT,
            data TEXT
        )
    ''')
    with get_db_connection() as conn:
        conn.execute(create_sql)
        conn.commit()

# db_utils.py
_db_initialized = False

def initialize_db():
    global _db_initialized
    if not _db_initialized:
        print("Initializing the database...")
        create_tables()
        _db_initialized = True
    # else:
    #     print("Database already initialized.")

def save_entity(entity_type, entity_id, data):
    print("Started Save Entity")
    upsert_sql = text('''
        INSERT INTO entities (id, entity_type, data)
        VALUES (:id, :etype, :data)
        ON CONFLICT(id) DO UPDATE
            SET entity_type = EXCLUDED.entity_type,
                data = EXCLUDED.data
    ''')
    select_sql = text('SELECT COUNT(*) FROM entities WHERE id = :id')

    try:
        with get_db_connection() as conn:
            # Check if the entity already exists
            result = conn.execute(select_sql, {"id": entity_id}).scalar()
            is_update = result > 0

            # Perform the upsert
            conn.execute(
                upsert_sql,
                {
                    "id": entity_id,
                    "etype": entity_type,
                    "data": json.dumps(data),
                }
            )
            conn.commit()

        # Construct the success response
        if is_update:
            return {
                "status": "success",
                "action": "updated",
                "id": entity_id,
                "entity_type": entity_type,
                "message": f"The {entity_type} with ID {entity_id} was successfully updated."
            }
        else:
            return {
                "status": "success",
                "action": "created",
                "id": entity_id,
                "entity_type": entity_type,
                "message": f"A new {entity_type} with ID {entity_id} was successfully created."
            }
    except Exception as e:
        # Construct the failure response
        print(f"Error while saving the entity with ID {entity_id}: ", {e})
        return {
            "status": "failure",
            "id": entity_id,
            "entity_type": entity_type,
            "message": f"An error occurred while saving the {entity_type} with ID {entity_id}.",
            "error": str(e)
        }
    
def load_entities(entity_type):
    query = text('SELECT id, data FROM entities WHERE entity_type = :etype')
    with get_db_connection() as conn:
        result = conn.execute(query, {"etype": entity_type})
        # result.mappings() gives us rows as dicts (if using SQLAlchemy 1.4+)
        rows = result.mappings().all()
    return [(row["id"], json.loads(row["data"])) for row in rows]
# def load_entities(entity_type):
#     query = text('SELECT id, data FROM entities WHERE entity_type = :etype')
    
#     try:
#         with get_db_connection() as conn:
#             result = conn.execute(query, {"etype": entity_type})
#             rows = result.mappings().all()

#         if not rows:
#             return {
#                 "status": "success",
#                 "message": f"No entities of type '{entity_type}' were found.",
#                 "entities": []
#             }

#         entities = [(row["id"], json.loads(row["data"])) for row in rows]
#         return {
#             "status": "success",
#             "message": f"{len(entities)} entities of type '{entity_type}' were successfully loaded.",
#             "entities": entities
#         }

#     except Exception as e:
#         return {
#             "status": "failure",
#             "message": f"An error occurred while attempting to load entities of type '{entity_type}'.",
#             "entity_type": entity_type,
#             "error": str(e)
#         }

def delete_entity(entity_type, entity_id):
    delete_sql = text('''
        DELETE FROM entities
        WHERE id = :id AND entity_type = :etype
    ''')
    select_sql = text('SELECT COUNT(*) FROM entities WHERE id = :id AND entity_type = :etype')
    
    try:
        with get_db_connection() as conn:
            # Check if the entity exists before attempting to delete
            exists = conn.execute(select_sql, {"id": entity_id, "etype": entity_type}).scalar() > 0

            if not exists:
                return {
                    "status": "failure",
                    "message": f"No {entity_type} with ID {entity_id} was found.",
                    "id": entity_id,
                    "entity_type": entity_type
                }

            # Perform the delete operation
            conn.execute(delete_sql, {"id": entity_id, "etype": entity_type})
            conn.commit()

        return {
            "status": "success",
            "message": f"The {entity_type} with ID {entity_id} was successfully deleted.",
            "id": entity_id,
            "entity_type": entity_type
        }

    except Exception as e:
        # Handle any unexpected errors
        return {
            "status": "failure",
            "message": f"An error occurred while attempting to delete the {entity_type} with ID {entity_id}.",
            "id": entity_id,
            "entity_type": entity_type,
            "error": str(e)
        }

def save_tools_state(enabled_tools):
    data = {
        'enabled_tools': enabled_tools
    }
    return save_entity('tools_state', 'enabled_tools', data)

def load_tools_state():
    rows = load_entities('tools_state')
    if rows:
        return rows[0][1].get('enabled_tools', {})
    return {}

def save_agent(agent):
    print("Started Save Agent")
    created_at = agent.created_at if hasattr(agent, 'created_at') and agent.created_at else datetime.utcnow().isoformat()
    data = {
        'created_at': created_at,
        'role': agent.role,
        'goal': agent.goal,
        'backstory': agent.backstory,
        'cache': agent.cache,
        'config': agent.config,
        'verbose': agent.verbose,
        'max_rpm': agent.max_rpm,
        'allow_delegation': agent.allow_delegation,
        'tool_ids': [tool.tool_id for tool in agent.tools],
        'max_iter': agent.max_iter,
        'llm_provider_model': agent.llm_provider_model,
        'temperature': agent.temperature,
        'max_tokens': agent.max_tokens
    }
    return save_entity('agent', agent.id, data)

def load_agents():
    from my_agent import MyAgent
    rows = load_entities('agent')
    tools_dict = {tool.tool_id: tool for tool in load_tools()}
    agents = []
    for row in rows:
        data = row[1]
        tool_ids = data.pop('tool_ids', [])
        agent = MyAgent(id=row[0], **data)
        agent.tools = [tools_dict[tool_id] for tool_id in tool_ids if tool_id in tools_dict]
        agents.append(agent)
    return sorted(agents, key=lambda x: x.created_at)

def get_agents_by_attributes(filters: dict) -> list:
    """
    Query the database for agents based on provided attributes.
    
    :param filters: A dictionary of attributes to filter by (e.g., {"role": "Statistician", "backstory": "An expert in baseball statistics"}).
    :return: A list of matching agents as dictionaries.
    """
    # Base SQL query
    query = "SELECT * FROM agents"
    
    # Dynamically build WHERE clause based on provided filters
    where_clauses = []
    params = {}
    for key, value in filters.items():
        where_clauses.append(f"{key} = :{key}")
        params[key] = value
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    query += " LIMIT 100"  # Optional limit for large results

    # Execute query
    with get_db_connection() as conn:
        result = conn.execute(query, params).fetchall()
        return [dict(row) for row in result]

def delete_agent(agent_id):
    return delete_entity('agent', agent_id)

def save_task(task):
    print("save_task:", task)
    print("The task id is: ", task.id)

    # Resolve agent
    agent = (
        task.agent if isinstance(task.agent, str)  # If `agent` is a string (ID)
        else task.agent.get("id") if isinstance(task.agent, dict) and "id" in task.agent  # If `agent` is a dict with "id"
        else getattr(task.agent, "id", None) if hasattr(task.agent, "id")  # If `agent` is an object with "id"
        else None
    )

    if not agent:
        raise HTTPException(status_code=400, detail=f"Could not resolve agent for task: {task.agent}")

    print(f"Resolved agent: {agent}")

    data = {
        'description': task.description,
        'expected_output': task.expected_output,
        'async_execution': task.async_execution,
        'agent': agent,  # Save `agent` directly, not `agent_id`
        'context_from_async_tasks_ids': task.context_from_async_tasks_ids,
        'context_from_sync_tasks_ids': task.context_from_sync_tasks_ids,
        'created_at': task.created_at,
    }
    return save_entity('task', task.id, data)

# def save_task(task):
#     print("save_task:", task)
#     print("The task id is: ", task.id)
#     agent_id = (
#         task.agent if isinstance(task.agent, str)  # If `agent` is a string (ID)
#         else task.agent.get("id") if isinstance(task.agent, dict) and "id" in task.agent  # If `agent` is a dict with "id"
#         else getattr(task.agent, "id", None) if hasattr(task.agent, "id")  # If `agent` is an object with "id"
#         else (get_agents_by_attributes({"backstory": task.agent.backstory})[0]["id"]  # Query using backstory
#             if hasattr(task.agent, "backstory") and task.agent.backstory 
#             else None)
#     )

#     if not agent_id:
#         raise HTTPException(status_code=400, detail=f"Could not resolve agent ID for task: {task.agent}")

#     print(f"Resolved agent_id: {agent_id}")

#     data = {
#         'description': task.description,
#         'expected_output': task.expected_output,
#         'async_execution': task.async_execution,
#         'agent_id': agent_id,
#         'context_from_async_tasks_ids': task.context_from_async_tasks_ids,
#         'context_from_sync_tasks_ids': task.context_from_sync_tasks_ids,
#         'created_at': task.created_at
#     }
#     return save_entity('task', task.id, data)

def load_tasks():
    from my_task import MyTask
    rows = load_entities('task')
    agents_dict = {agent.id: agent for agent in load_agents()}
    tasks = []

    for row in rows:
        data = row[1]
        agent_id = data.pop('agent_id', None)  # Extract agent_id from data
        if 'agent' in data:  # Remove 'agent' key if it exists in the data
            data.pop('agent')
        task = MyTask(id=row[0], agent=agents_dict.get(agent_id), **data)
        tasks.append(task)

    return sorted(tasks, key=lambda x: x.created_at)


def delete_task(task_id):
    return delete_entity('task', task_id)

def save_crew(crew):
    data = {
        'id': crew.id,
        'name': crew.name,
        'process': crew.process,
        'verbose': crew.verbose,
        'agent_ids': [agent.id for agent in crew.agents],
        'agent': [agent.id for agent in crew.agents],
        'task_ids': [task.id for task in crew.tasks],
        # 'agents': crew.agents,
        # 'tasks': crew.tasks,
        'memory': crew.memory,
        'cache': crew.cache,
        'planning': crew.planning,
        'max_rpm': crew.max_rpm,
        'manager_llm': crew.manager_llm,
        'manager_agent_id': crew.manager_agent.id if crew.manager_agent else None,
        'function_calling_llm': crew.function_calling_llm,
        'config': crew.config,
        'prompt_file': crew.prompt_file,
        'memory_config': crew.memory_config,
        'task_callback': str(crew.task_callback) if crew.task_callback else None,
        'step_callback': str(crew.step_callback) if crew.step_callback else None,
        'share_crew': crew.share_crew if crew.share_crew else False,
        'created_at': crew.created_at
    }
    print(f"Saving crew {crew.id} ...{data}")
    return save_entity('crew', crew.id, data)

def load_crews():
    from my_crew import MyCrew
    rows = load_entities('crew')
    agents_dict = {agent.id: agent for agent in load_agents()}
    tasks_dict = {task.id: task for task in load_tasks()}
    crews = []

    for row in rows:
        try:
            data = row[1]
            # Validate presence of required keys
            required_keys = ['name', 'process', 'created_at']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key '{key}' in crew data: {data}")

            # Log and skip missing agents or tasks
            missing_agents = [agent for agent in data.get('agents', []) if agent not in agents_dict]
            if missing_agents:
                print(f"Warning: Missing agents {missing_agents} for crew {row[0]}")

            missing_tasks = [task_id for task_id in data.get('task_ids', []) if task_id not in tasks_dict]
            if missing_tasks:
                print(f"Warning: Missing tasks {missing_tasks} for crew {row[0]}")
            crew = MyCrew(
                id=row[0],
                name=data['name'],
                process=data['process'],
                verbose=data.get('verbose', True),  # Default to True if not provided
                created_at=data['created_at'],
                memory=data.get('memory', False),  # Default to False if not provided
                cache=data.get('cache', True),     # Default to True if not provided
                planning=data.get('planning', False),  # Default to False if not provided
                max_rpm=data.get('max_rpm', 1000),     # Default to 1000 if not provided
                manager_llm=data.get('manager_llm'),
                manager_agent=agents_dict.get(data.get('manager_agent_id')),
                function_calling_llm=data.get('function_calling_llm'),
                config=data.get('config', {}),  # Default to an empty dictionary
                prompt_file=data.get('prompt_file'),
                memory_config=data.get('memory_config', {}),  # Default to an empty dictionary
                task_callback=data.get('task_callback'),
                step_callback=data.get('step_callback'),
                share_crew=data.get('share_crew', False)  # Default to False if not provided
            )
            crew.agents = [agents_dict[agent_id] for agent_id in data.get('agent_ids', []) if agent_id in agents_dict]
            crew.tasks = [tasks_dict[task_id] for task_id in data.get('task_ids', []) if task_id in tasks_dict]
            crews.append(crew)
        except Exception as e:
            # Log error and skip this crew
            print(f"Error loading crew {row[0]}: {str(e)}")
            continue

    return sorted(crews, key=lambda x: x.created_at)

def delete_crew(crew_id):
    return delete_entity('crew', crew_id)

def save_tool(tool):
    data = {
        'name': tool.name,
        'description': tool.description,
        'parameters': tool.get_parameters()
    }
    return save_entity('tool', tool.tool_id, data)

def load_tools():
    rows = load_entities('tool')
    tools = []
    for row in rows:
        data = row[1]
        tool_class = TOOL_CLASSES[data['name']]
        tool = tool_class(tool_id=row[0])
        tool.set_parameters(**data['parameters'])
        tools.append(tool)
    return tools

def delete_tool(tool_id):
    return delete_entity('tool', tool_id)

def export_to_json(file_path):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM entities')
    rows = cursor.fetchall()
    conn.close()

    data = []
    for row in rows:
        entity = {
            'id': row['id'],
            'entity_type': row['entity_type'],
            'data': json.loads(row['data'])
        }
        data.append(entity)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def import_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    conn = get_db_connection()
    cursor = conn.cursor()
    
    for entity in data:
        cursor.execute('''
            INSERT OR REPLACE INTO entities (id, entity_type, data)
            VALUES (?, ?, ?)
        ''', (entity['id'], entity['entity_type'], json.dumps(entity['data'])))

    conn.commit()
    conn.close()

# def get_agent_by_id(agent_id: str) -> Optional[dict]:
#     print("agent_id: ", agent_id)
#     """
#     Retrieve an agent from the database by ID.
    
#     :param agent_id: The ID of the agent to retrieve.
#     :return: A dictionary representing the agent or None if not found.
#     """
#     # query = "SELECT * FROM agents WHERE id = :id LIMIT 1"
#     query = text('SELECT id, data FROM entities WHERE entity_type = "agent" AND id = :id LIMIT 1')

#     with get_db_connection() as conn:
#         result = conn.execute(query, {"id": agent_id}).fetchone()
#         return dict(result) if result else None

# def get_task_by_id(task_id: str) -> Optional[dict]:
#     print("task_id: ", task_id)
#     """
#     Retrieve a task from the database by ID.
    
#     :param task_id: The ID of the task to retrieve.
#     :return: A dictionary representing the task or None if not found.
#     """
#     # query = text('SELECT * FROM tasks WHERE id = :id LIMIT 1')
#     query = text('SELECT id, data FROM entities WHERE entity_type = "task" AND id = :id LIMIT 1')

#     with get_db_connection() as conn:
#         result = conn.execute(query, {"id": task_id}).fetchone()
#         return dict(result) if result else None

def get_entity_by_id(entity_type: str, entity_id: str):
    print(f"Fetch entity, {entity_type} with ID {entity_id}.")
    """
    Fetch an entity by its type and ID.
    :param entity_type: The type of the entity (e.g., 'agent', 'task', 'crew').
    :param entity_id: The ID of the entity.
    :return: The entity data as a dictionary if found, else None.
    """
    # query = text('SELECT id, data FROM entities WHERE entity_type = :etype AND id = :id LIMIT 1')
    query = text('SELECT data FROM entities WHERE entity_type = :etype AND id = :id LIMIT 1')
    try:
        with get_db_connection() as conn:
            result = conn.execute(query, {"etype": entity_type, "id": entity_id}).fetchone()

            if not result:
                return None
            entity_data = result["data"] if isinstance(result, dict) else result[0]

            # Parse the JSON data
            parsed_data = json.loads(entity_data)
            parsed_data["id"] = entity_id  # Add the ID back to the parsed data
            print(f"Parsed {entity_type} data: {parsed_data}")

            if entity_type == "task" and "agent_id" in parsed_data:
                # Map agent_id to agent for compatibility with TaskPayload
                parsed_data["agent"] = parsed_data.pop("agent_id")
            
            print(f"Re-Parsed {entity_type} data: {parsed_data}")

            # if not result["data"]:
            #     print("There is no data column for this entity")
            #     return None  # Entity not found
            # # Access result as a tuple
            # print(f"Entity data: {result["data"]}")
            # # entity_id, entity_data = result
            # # parsed_data = json.loads(entity_data)
            # # parsed_data["id"] = entity_id
            # parsed_data = json.loads(result["data"])
            # parsed_data["id"] = entity_id
            # print(f"Parsed {entity_type} data: {parsed_data}")
            
            # if entity_type == "agent":
            #     try:
            #         validated_agent = AgentPayload(**parsed_data)
            #         return {validated_agent}
            #     except Exception as e:
            #         print(f"Validation error: {e}")
            #         raise HTTPException(status_code=500, detail="Failed to validate agent.")
            # elif entity_type == "task":
            #     try:
            #         validated_task = TaskPayload(**parsed_data)
            #         return {validated_task}
            #     except Exception as e:
            #         print(f"Validation error: {e}")
            #         raise HTTPException(status_code=500, detail="Failed to validate task.")
            # elif entity_type == "crew":
            #     try:
            #         validated_crew = CrewPayload(**parsed_data)
            #         return {validated_crew}
            #     except Exception as e:
            #         print(f"Validation error: {e}")
            #         raise HTTPException(status_code=500, detail="Failed to validate crew.")
            return parsed_data
    except Exception as e:
        print(f"Error fetching entity of type {entity_type} with ID {entity_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch entity.")