from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pydantic.error_wrappers import ValidationError
from fastapi.exceptions import RequestValidationError
from typing import Optional, List, Union
import db_utils
import uuid
from fastapi.responses import JSONResponse
from datetime import datetime
from my_crew import MyCrew
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
router = APIRouter()

# Pydantic Models for Validation
class EntityPayload(BaseModel):
    entity_type: str
    entity_id: str
    data: dict

class AgentPayload(BaseModel):
    id: Optional[str] = None
    role: str
    backstory: Optional[str] = None
    goal: str
    cache: bool
    temperature: float
    allow_delegation: bool
    verbose: bool
    llm_provider_model: str
    created_at: Optional[str] = None
    tools: Optional[List[dict]] = []
    max_iter: Optional[int] = None
    config: Optional[dict] = None
    max_rpm: Optional[int] = None
    max_tokens: Optional[int] = None

    def is_valid(self, show_warning: bool = False) -> bool:
        if not self.id:
            if show_warning:
                print("Validation Error: Agent must have an ID.")
            return False

        if not self.role:
            if show_warning:
                print("Validation Error: Agent must have a role.")
            return False

        if not self.goal:
            if show_warning:
                print("Validation Error: Agent must have a goal.")
            return False

        if not (0 <= self.temperature <= 1):
            if show_warning:
                print(f"Validation Error: Temperature must be between 0 and 1. Got {self.temperature}.")
            return False

        if self.max_rpm is not None and (self.max_rpm <= 0 or not isinstance(self.max_rpm, int)):
            if show_warning:
                print(f"Validation Error: max_rpm must be a positive integer. Got {self.max_rpm}.")
            return False

        if self.max_tokens is not None and (self.max_tokens <= 0 or self.max_tokens > 4096):
            if show_warning:
                print(f"Validation Error: max_tokens must be between 1 and 4096. Got {self.max_tokens}.")
            return False

        if self.config and not isinstance(self.config, dict):
            if show_warning:
                print("Validation Error: Config must be a dictionary.")
            return False

        if self.tools and not all(isinstance(tool, dict) for tool in self.tools):
            if show_warning:
                print("Validation Error: All tools must be dictionaries.")
            return False

        return True

class TaskPayload(BaseModel):
    id: Optional[str] = None
    description: str
    expected_output: str
    async_execution: bool
    agent: Optional[str]  # Could be an ID or reference
    context_from_async_tasks_ids: List[str] = []
    context_from_sync_tasks_ids: List[str] = []
    created_at: Optional[str] = None

    def is_valid(self, agents: Optional[List[AgentPayload]] = None, show_warning: bool = False) -> bool:
        if not self.id:
            if show_warning:
                print("Validation Error: Task must have an ID.")
            return False

        if not self.description:
            if show_warning:
                print("Validation Error: Task must have a description.")
            return False

        if not self.expected_output:
            if show_warning:
                print("Validation Error: Task must have an expected output.")
            return False

        if self.async_execution and not isinstance(self.async_execution, bool):
            if show_warning:
                print(f"Validation Error: async_execution must be a boolean. Got {self.async_execution}.")
            return False

        if self.agent:
            if not isinstance(self.agent, str):
                if show_warning:
                    print(f"Validation Error: Agent must be a valid string. Got {self.agent}.")
                return False
            # Check if the agent ID exists in the provided list of agents
            if agents and self.agent not in [agent.id for agent in agents]:
                if show_warning:
                    print(f"Validation Error: Agent ID '{self.agent}' does not exist.")
                return False

        if not all(isinstance(context_id, str) for context_id in self.context_from_async_tasks_ids):
            if show_warning:
                print("Validation Error: All async context IDs must be strings.")
            return False

        if not all(isinstance(context_id, str) for context_id in self.context_from_sync_tasks_ids):
            if show_warning:
                print("Validation Error: All sync context IDs must be strings.")
            return False

        return True

class CrewPayload(BaseModel):
    id: Optional[str] = None
    name: str
    process: Optional[str] = None
    verbose: bool
    agents: Optional[List[Union[str, AgentPayload]]] = None
    tasks: Optional[List[Union[str, TaskPayload]]] = None
    memory: bool
    cache: bool
    planning: bool
    max_rpm: Optional[int] = None
    manager_llm: Optional[str] = None
    manager_agent: Optional[str] = None
    created_at: Optional[str] = None
    function_calling_llm: Optional[str] = None
    config: Optional[dict] = None
    prompt_file: Optional[str] = None
    memory_config: Optional[dict] = None

    def is_valid(self, agents: Optional[List[AgentPayload]] = None, show_warning: bool = False) -> bool:
        if not self.name:
            if show_warning:
                print("Validation Error: Crew must have a name.")
            return False

        if self.process not in ["sequential", "hierarchical"]:
            if show_warning:
                print(f"Validation Error: Invalid process '{self.process}'.")
            return False

        for agent in self.agents or []:
            if isinstance(agent, AgentPayload) and not agent.is_valid(show_warning=show_warning):
                return False

        for task in self.tasks or []:
            if isinstance(task, TaskPayload) and not task.is_valid(agents=agents, show_warning=show_warning):
                return False

        if self.process == "hierarchical" and not (self.manager_llm or self.manager_agent):
            if show_warning:
                print("Validation Error: Hierarchical process requires either a manager_llm or manager_agent.")
            return False

        return True

# General Entity Routes
@router.get("/entities/{entity_type}")
def load_entities(entity_type: str):
    return {"entities": db_utils.load_entities(entity_type)}

@router.post("/entities/")
def save_entity(payload: EntityPayload):
    if not payload.id:
        payload.entity_id = str(uuid.uuid4())
    response=db_utils.save_entity(payload.entity_type, payload.entity_id, payload.data)
    # return {"status": "success", "entity_type": payload.entity_type, "entity_id": payload.entity_id}
    if response["status"] == "failure":
        raise HTTPException(status_code=500, detail=response)
    return response

@router.delete("/entities/{entity_type}/{entity_id}")
def delete_entity(entity_type: str, entity_id: str):
    response = db_utils.delete_entity(entity_type, entity_id)
    # return {"status": "success", "entity_type": entity_type, "entity_id": entity_id}
    if response["status"] == "failure":
        raise HTTPException(status_code=404 if "No" in response["message"] else 500, detail=response)
    return response

# Agent Routes
@router.get("/agents/")
def load_agents():
    return {"agents": db_utils.load_agents()}

@router.post("/agents/")
def save_agent(payload: AgentPayload):
    try:
        # Auto-generate an ID if not provided
        if not payload.id:
            payload.id = str(uuid.uuid4())
        print("Received Payload:", payload.model_dump())
        response = db_utils.save_agent(payload)
        if response["status"] == "failure":
            raise HTTPException(status_code=500, detail=response)
        return response
    except Exception as e:
        return {
            "status": "failure",
            "error": str(e)
        }

@router.delete("/agents/{agent_id}")
def delete_agent(agent_id: str):
    response = db_utils.delete_agent(agent_id)
    if response["status"] == "failure":
        raise HTTPException(status_code=404 if "No" in response["message"] else 500, detail=response)
    return response

# Task Routes
@router.get("/tasks/")
def load_tasks():
    return {"tasks": db_utils.load_tasks()}

@router.post("/tasks/")
def save_task(payload: TaskPayload):
    try:
        # Auto-generate an ID if not provided
        if not payload.id:
            payload.id = str(uuid.uuid4())
        print("Task Payload Received:", payload.model_dump())
        response =db_utils.save_task(payload)
        # return {"status": "success", "task_id": payload.id}
        if response["status"] == "failure":
            raise HTTPException(status_code=500, detail=response)
        return response
    except Exception as e:
        return {
            "status": "failure",
            "error": str(e)
        }

@router.delete("/tasks/{task_id}")
def delete_task(task_id: str):
    response = db_utils.delete_task(task_id)
    # return {"status": "success", "task_id": task_id}
    if response["status"] == "failure":
        raise HTTPException(status_code=404 if "No" in response["message"] else 500, detail=response)
    return response

# Crew Routes
@router.get("/crews/")
def load_crews():
    return {"crews": db_utils.load_crews()}

@router.post("/crews/")
def create_crew(payload: CrewPayload):
    try:
        print("Reached create_crew endpoint")
        # print(f"Payload: {payload.dict()}")
        print("Received Crew Payload:", payload.model_dump())

        # Process agents
        agents = []
        for agent in payload.agents or []:
            if isinstance(agent, str):  # If it's an ID
                print(f"Agent ID provided: {agent}")
                agents.append({"id": agent})
            elif isinstance(agent, AgentPayload):  # If it's already an AgentPayload instance
                print(f"AgentPayload instance provided: {agent}")
                save_response = save_agent(agent)  # Pass it directly
                print(f"Agent Save response: {save_response}")
                if save_response["status"] != "success":
                    raise HTTPException(status_code=500, detail=f"Failed to save agent: {save_response.get('error')}")
                agents.append({"id": save_response["id"]})
            elif isinstance(agent, dict):  # If it's a dictionary
                print(f"Full agent object (dict) provided: {agent}")
                save_response = save_agent(AgentPayload(**agent))  # Convert dict to AgentPayload
                print(f"Save response: {save_response}")
                if save_response["status"] != "success":
                    raise HTTPException(status_code=500, detail=f"Failed to save agent: {save_response.get('error')}")
                agents.append({"id": save_response["id"]})
            else:
                logging.error(f"Invalid agent format: {agent}")
                raise HTTPException(status_code=400, detail="Invalid agent format. Must be an ID, AgentPayload, or a valid agent object.")
        print("Start Tasks")
        # Process tasks
        tasks = []
        for task in payload.tasks or []:
            if isinstance(task, str):  # If it's an ID
                print(f"Task ID provided: {task}")
                tasks.append({"id": task})
            elif isinstance(task, TaskPayload):  # If it's already a TaskPayload instance
                print(f"TaskPayload instance provided: {task}")
                save_response = save_task(task)  # Pass it directly
                print(f"Task Save response: {save_response}")
                if save_response["status"] != "success":
                    raise HTTPException(status_code=500, detail=f"Failed to save task: {save_response.get('error')}")
                tasks.append({"id": save_response["id"]})
            elif isinstance(task, dict):  # If it's a dictionary
                print(f"Full task object (dict) provided: {task}")
                save_response = save_task(TaskPayload(**task))  # Convert dict to TaskPayload
                if save_response["status"] != "success":
                    raise HTTPException(status_code=500, detail=f"Failed to save task: {save_response.get('error')}")
                tasks.append({"id": save_response["id"]})  # Append the ID
            else:
                print(f"Invalid task format: {task}")
                raise HTTPException(status_code=400, detail="Invalid task format. Must be an ID, TaskPayload, or a valid task object.")

        print("Let's load the agents")
        print(f"Agents: {agents}")
        print(f"Tasks: {tasks}")
        # Load agents based on their IDs
        agents_payloads = []
        for agent in agents:
            try:
                # Check if the agent matches the AgentPayload structure
                if isinstance(agent, AgentPayload):
                    agents_payloads.append(agent)
                elif isinstance(agent, dict) and "id" in agent:
                    agent_data = db_utils.get_entity_by_id("agent", agent["id"])
                    if not agent_data:
                        raise HTTPException(status_code=404, detail=f"Agent with ID '{agent['id']}' not found.")
                    agents_payloads.append(AgentPayload(**agent_data))
                elif isinstance(agent, str):  # If it's just an ID
                    agent_data = db_utils.get_entity_by_id("agent", agent)
                    # agent_data = db_utils.get_agent_by_id(agent)
                    if not agent_data:
                        raise HTTPException(status_code=404, detail=f"Agent with ID '{agent}' not found.")
                    agents_payloads.append(AgentPayload(**agent_data))
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid agent format. Must be an ID, dictionary with 'id', or valid AgentPayload object."
                    )
            # except ValidationError as e:
            #     raise HTTPException(
            #         status_code=422,
            #         detail=f"Validation error for agent: {e.errors()}"
            #     )
            # except ValidationError as e:
            #     raise RequestValidationError(e.raw_errors)
            except Exception as e:
                print(f"Error processing agent: {agent}, Exception: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to fetch agents.")

        # Load tasks based on their IDs
        tasks_payloads = []
        # for task in tasks:
        #     try:
        #         # Check if the task matches the TaskPayload structure
        #         if isinstance(task, TaskPayload):
        #             tasks_payloads.append(task)
        #         elif isinstance(task, dict) and "id" in task:
        #             task_data = db_utils.get_entity_by_id("task", task["id"])
        #             if not task_data:
        #                 raise HTTPException(status_code=404, detail=f"Task with ID '{task['id']}' not found.")
        #             tasks_payloads.append(TaskPayload(**task_data))
        #         elif isinstance(task, str):  # If it's just an ID
        #             # task_data = db_utils.get_task_by_id(task)
        #             task_data = db_utils.get_entity_by_id("task", task)
        #             if not task_data:
        #                 raise HTTPException(status_code=404, detail=f"Task with ID '{task}' not found.")
        #             tasks_payloads.append(TaskPayload(**task_data))
        #         else:
        #             raise HTTPException(
        #                 status_code=400,
        #                 detail="Invalid task format. Must be an ID, dictionary with 'id', or valid TaskPayload object."
        #             )
        #     except Exception as e:
        #         print(f"Error processing task: {task}, Exception: {str(e)}")
        #         raise HTTPException(status_code=500, detail="Failed to fetch tasks.")
        for task in tasks:
            try:
                if isinstance(task, TaskPayload):
                    tasks_payloads.append(task)
                elif isinstance(task, dict) and "id" in task:
                    task_data = db_utils.get_entity_by_id("task", task["id"])
                    if not task_data:
                        raise HTTPException(status_code=404, detail=f"Task with ID '{task['id']}' not found.")
                    
                    # Ensure `agent_id` is mapped to `agent`
                    if "agent_id" in task_data:
                        task_data["agent"] = task_data.pop("agent_id")
                    tasks_payloads.append(TaskPayload(**task_data))
                elif isinstance(task, str):  # If it's just an ID
                    task_data = db_utils.get_entity_by_id("task", task)
                    if not task_data:
                        raise HTTPException(status_code=404, detail=f"Task with ID '{task}' not found.")
                    
                    # Ensure `agent_id` is mapped to `agent`
                    if "agent_id" in task_data:
                        task_data["agent"] = task_data.pop("agent_id")
                    tasks_payloads.append(TaskPayload(**task_data))
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid task format. Must be an ID, dictionary with 'id', or valid TaskPayload object."
                    )
            except Exception as e:
                logging.error(f"Error processing task: {task}, Exception: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to fetch tasks.")




        print("Agents payloads: ", agents_payloads)
        print("Tasks payloads: ", tasks_payloads)
        # agents=[agent.dict() for agent in agents_payloads]
        # tasks=[task.dict() for task in tasks_payloads]
        # print("Final Agents: ", agents)
        # print("Final Tasks: ", tasks)
        # Initialize MyCrew instance
        try:
            print("===== DEBUG: Payload Details =====")
            print(f"Payload ID: {payload.id}")
            print(f"Payload Name: {payload.name}")
            print(f"Payload Process: {payload.process}")
            print(f"Payload Verbose: {payload.verbose}")
            print(f"Payload Memory: {payload.memory}")
            print(f"Payload Cache: {payload.cache}")
            print(f"Payload Planning: {payload.planning}")
            print(f"Payload Max RPM: {payload.max_rpm}")
            print(f"Payload Manager LLM: {payload.manager_llm}")
            print(f"Payload Manager Agent: {payload.manager_agent}")
            print(f"Payload Created At: {payload.created_at}")
            print(f"Payload Function Calling LLM: {payload.function_calling_llm}")
            print(f"Payload Config: {payload.config}")
            print(f"Payload Prompt File: {payload.prompt_file}")
            print(f"Payload Memory Config: {payload.memory_config}")

            print("\n===== DEBUG: Agents and Tasks =====")
            print("Agents Loaded:")
            for agent in agents:
                print(agent)  # Log each agent object or dictionary

            print("\nTasks Loaded:")
            for task in tasks:
                print(task)  # Log each task object or dictionary

            # Initialize MyCrew instance
            crew = MyCrew(
                id=payload.id or f"C_{uuid.uuid4()}",
                name=payload.name,
                process=payload.process,
                verbose=payload.verbose,
                agents=agents_payloads,  # Pass validated agents
                tasks=tasks_payloads,    # Pass validated tasks
                memory=payload.memory,
                cache=payload.cache,
                planning=payload.planning,
                max_rpm=payload.max_rpm,
                manager_llm=payload.manager_llm,
                manager_agent=payload.manager_agent,
                created_at=payload.created_at or datetime.utcnow().isoformat(),
                function_calling_llm=payload.function_calling_llm,
                config=payload.config,
                prompt_file=payload.prompt_file,
                memory_config=payload.memory_config,
            )

            print("\n===== DEBUG: Crew Created =====")
            print(f"Crew ID: {crew.id}")
            print(f"Crew Name: {crew.name}")
            print(f"Crew Agents: {crew.agents}")
            print(f"Crew Tasks: {crew.tasks}")

            # Validate and save the crew
            if not crew.is_valid(show_warning=True):
                invalid_agents = [agent for agent in crew.agents if not agent.is_valid()]
                invalid_tasks = [task for task in crew.tasks if not task.is_valid()]
                raise ValueError(f"Invalid crew configuration. Invalid agents: {invalid_agents}, Invalid tasks: {invalid_tasks}")

            save_results = db_utils.save_crew(crew)
            print("\n===== DEBUG: Crew Saved =====\n")
            print(save_results)

            if (save_results["status"]!= "success"):
                return {
                    "status": "failure",
                    "id": "",
                    "message": f"Crew '{crew.name}' failed to create. {save_results["message"]} -> {save_results["error"]}"
                }
            
            crew_data = db_utils.get_entity_by_id("crew", crew.id)
            print("\n===== DEBUG: Crew Saved =====")
            print(crew_data)

            return {
                "status": "success",
                "id": crew.id,
                "message": f"Crew '{crew.name}' created successfully."
            }

        except Exception as e:
            print(f"Error during crew creation: {e}")
            return {
                "status": "failure",
                "status_code": 500,
                "id": None,
                "message": f"Crew creation failed. {str(e)}",
                "crew": {}
            }
    except Exception as e:
        print(f"Error during crew creation: {e}")
        return {
            "status": "failure",
            "status_code": 500,
            "id": None,
            "message": f"Crew creation failed. {str(e)}",
            "crew": {}
        }



@router.delete("/crews/{crew_id}")
def delete_crew(crew_id: str):
    response = db_utils.delete_crew(crew_id)
    # return {"status": "success", "crew_id": crew_id}
    if response["status"] == "failure":
        raise HTTPException(status_code=404 if "No" in response["message"] else 500, detail=response)
    return response

# Tool Routes
@router.get("/tools/")
def load_tools():
    return {"tools": db_utils.load_tools()}

@router.post("/tools/")
def save_tool(payload: dict):
    if not payload.id:
        payload.id = str(uuid.uuid4())
    response = db_utils.save_tool(payload)
    if response["status"] == "failure":
        raise HTTPException(status_code=500, detail=response)
    return response


@router.delete("/tools/{tool_id}")
def delete_tool(tool_id: str):
    response = db_utils.delete_tool(tool_id)
    # return {"status": "success", "tool_id": tool_id}
    if response["status"] == "failure":
        raise HTTPException(status_code=404 if "No" in response["message"] else 500, detail=response)
    return response

# Tools State
@router.get("/tools/state/")
def load_tools_state():
    return {"tools_state": db_utils.load_tools_state()}

@router.post("/tools/state/")
def save_tools_state(payload: dict):
    db_utils.save_tools_state(payload.get("enabled_tools"))
    return {"status": "success"}