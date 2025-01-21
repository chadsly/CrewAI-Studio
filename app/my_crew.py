from crewai import Crew, Process
import streamlit as st
from utils import rnd_id, fix_columns_width
from streamlit import session_state as ss
from datetime import datetime
from llms import llm_providers_and_models, create_llm
import db_utils
import json

class MyCrew:
    def __init__(self, id=None, name=None, process=None, verbose=None, agents=None, tasks=None, memory=None,
                 cache=None, planning=None, max_rpm=None, manager_llm=None, manager_agent=None, created_at=None,
                 function_calling_llm=None, config=None, prompt_file=None, memory_config=None,
                 task_callback=None, step_callback=None, share_crew=None):
        self.id = id or "C_" + rnd_id()
        self.name = name or "Crew 1"
        self.process = process or Process.sequential
        self.verbose = verbose if verbose is not None else True
        self.agents = agents or []
        self.tasks = tasks or []
        self.memory = memory if memory is not None else False
        self.cache = cache if cache is not None else True
        self.planning = planning if planning is not None else False
        self.max_rpm = max_rpm or 1000
        self.manager_llm = manager_llm
        self.manager_agent = manager_agent
        self.created_at = created_at or datetime.now().isoformat()
        self.function_calling_llm = function_calling_llm or ""  # Default to empty string
        self.config = config or {}  # Default to empty dictionary
        self.prompt_file = prompt_file or ""  # Default to empty string
        self.memory_config = memory_config or {}   # Default to empty dictionary
        self.task_callback = task_callback or (lambda *args, **kwargs: None)  # Default no-op function
        self.step_callback = step_callback or (lambda *args, **kwargs: None)  # Default no-op function
        self.share_crew = share_crew if share_crew is not None else False  # Default to False
        self.edit_key = f'edit_{self.id}'
        if self.edit_key not in ss:
            ss[self.edit_key] = False
        self.tasks_order_key = f'tasks_order_{self.id}'
        if self.tasks_order_key not in ss:
            ss[self.tasks_order_key] = [task.id for task in self.tasks]

    @property
    def edit(self):
        return ss[self.edit_key]

    @edit.setter
    def edit(self, value):
        ss[self.edit_key] = value

    def get_crewai_crew(self, *args, **kwargs) -> Crew:
        crewai_agents = [agent.get_crewai_agent() for agent in self.agents]

        # Create a dictionary to hold the Task objects
        task_objects = {}

        def create_task(task):
            if task.id in task_objects:
                return task_objects[task.id]

            context_tasks = []
            if task.async_execution or task.context_from_async_tasks_ids or task.context_from_sync_tasks_ids:
                for context_task_id in (task.context_from_async_tasks_ids or []) + (task.context_from_sync_tasks_ids or []):
                    if context_task_id not in task_objects:
                        context_task = next((t for t in self.tasks if t.id == context_task_id), None)
                        if context_task:
                            context_tasks.append(create_task(context_task))
                        else:
                            print(f"Warning: Context task with id {context_task_id} not found for task {task.id}")
                    else:
                        context_tasks.append(task_objects[context_task_id])

            # Only pass context if it's an async task or if specific context is defined
            if task.async_execution or context_tasks:
                crewai_task = task.get_crewai_task(context_from_async_tasks=context_tasks)
            else:
                crewai_task = task.get_crewai_task()

            task_objects[task.id] = crewai_task
            return crewai_task

        # Create all tasks, resolving dependencies recursively
        for task in self.tasks:
            create_task(task)

        # Collect the final list of tasks in the original order
        crewai_tasks = [task_objects[task.id] for task in self.tasks]

        return Crew(
            agents=crewai_agents,
            tasks=crewai_tasks,
            process=self.process,
            verbose=self.verbose,
            memory=self.memory,
            cache=self.cache,
            planning=self.planning,
            max_rpm=self.max_rpm,
            manager_llm=create_llm(self.manager_llm) if self.manager_llm else None,
            manager_agent=self.manager_agent.get_crewai_agent() if self.manager_agent else None,
            function_calling_llm=self.function_calling_llm,
            config=self.config,
            prompt_file=self.prompt_file,
            memory_config=self.memory_config,
            task_callback=self.task_callback,
            step_callback=self.step_callback,
            share_crew=self.share_crew,
            created_at=self.created_at,
            *args, **kwargs
        )
    
    def delete(self):
        ss.crews = [crew for crew in ss.crews if crew.id != self.id]
        db_utils.delete_crew(self.id)

    def update_name(self):
        self.name = ss[f'name_{self.id}']
        db_utils.save_crew(self)

    def update_process(self):
        self.process = ss[f'process_{self.id}']
        db_utils.save_crew(self)

    def update_tasks(self):
        selected_tasks_ids = ss[f'tasks_{self.id}']
        self.tasks = [task for task in ss.tasks if task.id in selected_tasks_ids and task.agent.id in [agent.id for agent in self.agents]]
        self.tasks = sorted(self.tasks, key=lambda task: selected_tasks_ids.index(task.id))
        ss[self.tasks_order_key] = selected_tasks_ids
        db_utils.save_crew(self)

    def update_verbose(self):
        self.verbose = ss[f'verbose_{self.id}']
        db_utils.save_crew(self)

    def update_agents(self):
        selected_agents = ss[f'agents_{self.id}']
        self.agents = [agent for agent in ss.agents if agent.role in selected_agents]        
        db_utils.save_crew(self)

    def update_manager_llm(self):
        selected_llm = ss[f'manager_llm_{self.id}']
        self.manager_llm = selected_llm if selected_llm != "None" else None
        if self.manager_llm:
            self.manager_agent = None
        db_utils.save_crew(self)

    def update_manager_agent(self):
        selected_agent_role = ss[f'manager_agent_{self.id}']
        self.manager_agent = next((agent for agent in ss.agents if agent.role == selected_agent_role), None) if selected_agent_role != "None" else None
        if self.manager_agent:
            self.manager_llm = None
        db_utils.save_crew(self)

    def update_memory(self):
        self.memory = ss[f'memory_{self.id}']
        db_utils.save_crew(self)
    
    def update_memory_config(self):
        try:
            # Attempt to parse the JSON string from Streamlit session state
            updated_config = json.loads(ss[f'memory_config_{self.id}'])
            self.memory_config = updated_config
            self.validate_memory_config()
            db_utils.save_crew(self)
        except json.JSONDecodeError:
            st.error("Invalid JSON format for memory_config.")
        except ValueError as e:
            st.error(str(e))

    
    def update_max_rpm(self):
        self.max_rpm = ss[f'max_rpm_{self.id}']
        db_utils.save_crew(self)
    
    def update_function_calling_llm(self):
        self.function_calling_llm = ss[f'function_calling_llm_{self.id}']
        db_utils.save_crew(self)
    
    def update_config(self):
        self.validate_config()
        self.config = ss[f'config_{self.id}']
        db_utils.save_crew(self)
    
    def update_prompt_file(self):
        self.prompt_file = ss[f'prompt_file_{self.id}']
        db_utils.save_crew(self)
    
    def update_task_callback(self):
        self.task_callback = ss[f'task_callback_{self.id}']
        db_utils.save_crew(self)
    
    def update_step_callback(self):
        self.step_callback = ss[f'step_callback_{self.id}']
        db_utils.save_crew(self)
    
    def update_share_crew(self):
        self.share_crew = ss[f'share_crew_{self.id}']
        db_utils.save_crew(self)

    def update_cache(self):
        self.cache = ss[f'cache_{self.id}']
        db_utils.save_crew(self)

    def update_planning(self):
        self.planning = ss[f'planning_{self.id}']
        db_utils.save_crew(self)

    def is_valid(self, show_warning=False):
        if len(self.agents) == 0:
            if show_warning:
                st.warning(f"Crew {self.name} has no agents")
            return False
        if len(self.tasks) == 0:
            if show_warning:
                st.warning(f"Crew {self.name} has no tasks")
            return False
        if any([not agent.is_valid(show_warning=show_warning) for agent in self.agents]):
            return False
        if any([not task.is_valid(show_warning=show_warning) for task in self.tasks]):
            return False
        if self.process == Process.hierarchical and not (self.manager_llm or self.manager_agent):
            if show_warning:
                st.warning(f"Crew {self.name} has no manager agent or manager llm set for hierarchical process")
            return False
        return True

    def validate_manager_llm(self):
        available_models = llm_providers_and_models()
        if self.manager_llm and self.manager_llm not in available_models:
            self.manager_llm = None

    def validate_config(self):
        """
        Performs lightweight validation of memory_config to ensure the format is correct and not harmful.
        """
        if not isinstance(self.memory_config, dict):
            raise ValueError("memory_config must be a dictionary.")

        # Validate nested 'config' key if it exists
        if "config" in self.memory_config:
            if not isinstance(self.memory_config["config"], dict):
                raise ValueError("The 'config' field in memory_config must be a dictionary.")
            
            # Optional: Check if the nested dictionary has valid keys/values
            for key, value in self.memory_config["config"].items():
                if not isinstance(key, str):
                    raise ValueError(f"Invalid key in memory_config['config']: {key}. Keys must be strings.")
                if not isinstance(value, (str, int, float, bool, type(None))):  # Allow basic JSON types
                    raise ValueError(f"Invalid value for key '{key}' in memory_config['config']: {value}.")
        
        # Optional: Limit size/depth to avoid abuse
        if len(self.memory_config) > 10:  # Limit top-level keys
            raise ValueError("memory_config has too many top-level keys.")
        if len(self.memory_config.get("config", {})) > 50:  # Limit keys in nested 'config'
            raise ValueError("memory_config['config'] has too many keys.")
        
    def validate_memory_config(self):
        """
        Performs lightweight validation of memory_config to ensure the format is correct and not harmful.
        """
        if not isinstance(self.memory_config, dict):
            raise ValueError("memory_config must be a dictionary.")

        # Validate nested 'config' key if it exists
        if "config" in self.memory_config:
            if not isinstance(self.memory_config["config"], dict):
                raise ValueError("The 'config' field in memory_config must be a dictionary.")
            
            # Optional: Check if the nested dictionary has valid keys/values
            for key, value in self.memory_config["config"].items():
                if not isinstance(key, str):
                    raise ValueError(f"Invalid key in memory_config['config']: {key}. Keys must be strings.")
                if not isinstance(value, (str, int, float, bool, type(None))):  # Allow basic JSON types
                    raise ValueError(f"Invalid value for key '{key}' in memory_config['config']: {value}.")
        
        # Optional: Limit size/depth to avoid abuse
        if len(self.memory_config) > 10:  # Limit top-level keys
            raise ValueError("memory_config has too many top-level keys.")
        if len(self.memory_config.get("config", {})) > 50:  # Limit keys in nested 'config'
            raise ValueError("memory_config['config'] has too many keys.")

    def draw(self, expanded=False, buttons=True):
        self.validate_manager_llm()
        self.validate_memory_config()

        self.function_calling_llm = self.function_calling_llm or ""
        self.config = self.config or {}
        self.prompt_file = self.prompt_file or ""
        self.memory_config = self.memory_config or {}
        self.task_callback = self.task_callback or (lambda *args, **kwargs: None)
        self.step_callback = self.step_callback or (lambda *args, **kwargs: None)
        self.share_crew = self.share_crew if self.share_crew is not None else False

        # Keys for Streamlit session state
        name_key = f"name_{self.id}"
        process_key = f"process_{self.id}"
        verbose_key = f"verbose_{self.id}"
        agents_key = f"agents_{self.id}"
        tasks_key = f"tasks_{self.id}"
        manager_llm_key = f"manager_llm_{self.id}"
        manager_agent_key = f"manager_agent_{self.id}"
        memory_key = f"memory_{self.id}"
        planning_key = f"planning_{self.id}"
        cache_key = f"cache_{self.id}"
        max_rpm_key = f"max_rpm_{self.id}"
        function_calling_llm_key = f"function_calling_llm_{self.id}"
        config_key = f"config_{self.id}"
        prompt_file_key = f"prompt_file_{self.id}"
        memory_config_key = f"memory_config_{self.id}"
        task_callback_key = f"task_callback_{self.id}"
        step_callback_key = f"step_callback_{self.id}"
        share_crew_key = f"share_crew_{self.id}"

        if self.edit:
            with st.container():
                st.text_input("Name (for display purposes)", value=self.name, key=name_key, on_change=self.update_name)
                st.selectbox(
                    "Process",
                    options=[Process.sequential, Process.hierarchical],
                    index=[Process.sequential, Process.hierarchical].index(self.process),
                    key=process_key,
                    on_change=self.update_process
                )
                st.multiselect(
                    "Agents",
                    options=[agent.role for agent in ss.agents],
                    default=[agent.role for agent in self.agents],
                    key=agents_key,
                    on_change=self.update_agents
                )
                # Filter tasks by selected agents
                available_tasks = [task for task in ss.tasks if task.agent and task.agent.id in [agent.id for agent in self.agents]]
                available_task_ids = [task.id for task in available_tasks]
                default_task_ids = [task.id for task in self.tasks if task.id in available_task_ids]
                st.multiselect(
                    "Tasks",
                    options=available_task_ids,
                    default=default_task_ids,
                    format_func=lambda x: next(task.description for task in ss.tasks if task.id == x),
                    key=tasks_key,
                    on_change=self.update_tasks
                )
                st.selectbox(
                    "Manager LLM",
                    options=["None"] + llm_providers_and_models(),
                    index=0 if self.manager_llm is None else llm_providers_and_models().index(self.manager_llm) + 1,
                    key=manager_llm_key,
                    on_change=self.update_manager_llm,
                    disabled=(self.process != Process.hierarchical)
                )
                st.selectbox(
                    "Manager Agent",
                    options=["None"] + [agent.role for agent in ss.agents],
                    index=0 if self.manager_agent is None else [agent.role for agent in ss.agents].index(self.manager_agent.role) + 1,
                    key=manager_agent_key,
                    on_change=self.update_manager_agent,
                    disabled=(self.process != Process.hierarchical)
                )
                st.checkbox("Verbose", value=self.verbose, key=verbose_key, on_change=self.update_verbose)
                st.checkbox("Memory", value=self.memory, key=memory_key, on_change=self.update_memory)
                st.checkbox("Cache", value=self.cache, key=cache_key, on_change=self.update_cache)
                st.checkbox("Planning", value=self.planning, key=planning_key, on_change=self.update_planning)
                st.number_input("Max Requests per Minute", value=self.max_rpm, min_value=1, key=max_rpm_key, on_change=self.update_max_rpm)

                # New attributes
                st.text_input("Function Calling LLM", value=self.function_calling_llm or "", key=function_calling_llm_key, on_change=self.update_function_calling_llm)
                st.text_area("Config (JSON)", value=str(self.config), key=config_key, on_change=self.update_config)
                st.text_input("Prompt File", value=self.prompt_file or "", key=prompt_file_key, on_change=self.update_prompt_file)
                st.text_area(
                    "Memory Config (JSON)",
                    value=json.dumps(self.memory_config, indent=2),
                    key=memory_config_key,
                    on_change=self.update_memory_config
                )
                st.text_input("Task Callback", value=str(self.task_callback) if self.task_callback else "", key=task_callback_key, on_change=self.update_task_callback)
                st.text_input("Step Callback", value=str(self.step_callback) if self.step_callback else "", key=step_callback_key, on_change=self.update_step_callback)
                st.checkbox("Share Crew", value=self.share_crew, key=share_crew_key, on_change=self.update_share_crew)

                st.button("Save", on_click=self.set_editable, args=(False,), key=rnd_id())
        else:
            fix_columns_width()
            expander_title = f"Crew: {self.name}" if self.is_valid() else f"❗ Crew: {self.name}"
            with st.expander(expander_title, expanded=expanded):
                st.markdown(f"**Process:** {self.process}")
                if self.process == Process.hierarchical:
                    st.markdown(f"**Manager LLM:** {self.manager_llm or 'None'}")
                    st.markdown(f"**Manager Agent:** {self.manager_agent.role if self.manager_agent else 'None'}")
                st.markdown(f"**Verbose:** {self.verbose}")
                st.markdown(f"**Memory:** {self.memory}")
                st.markdown(f"**Cache:** {self.cache}")
                st.markdown(f"**Planning:** {self.planning}")
                st.markdown(f"**Max Requests per Minute:** {self.max_rpm}")

                # Display new attributes
                st.markdown(f"**Function Calling LLM:** {self.function_calling_llm or 'None'}")
                st.markdown(f"**Config:** {self.config}")
                st.markdown(f"**Prompt File:** {self.prompt_file or 'None'}")
                st.markdown(f"**Memory Config:** {self.memory_config}")
                st.markdown(f"**Task Callback:** {self.task_callback or 'None'}")
                st.markdown(f"**Step Callback:** {self.step_callback or 'None'}")
                st.markdown(f"**Share Crew:** {'Yes' if self.share_crew else 'No'}")

                st.markdown("**Tasks:**")
                for i, task in enumerate([task for task in self.tasks if task.agent and task.agent.id in [agent.id for agent in self.agents]], 1):
                    with st.container():
                        async_tag = "(async)" if task.async_execution else ""
                        st.markdown(f"**{i}.{async_tag}  {task.description}**")
                        st.markdown(f"**Agent:** {task.agent.role if task.agent else 'None'}")
                        tools_list = ", ".join([tool.name for tool in task.agent.tools]) if task.agent else "None"
                        st.markdown(f"**Tools:** {tools_list}")
                        st.markdown(f"**LLM:** {task.agent.llm_provider_model}")
                if buttons:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.button("Edit", on_click=self.set_editable, args=(True,), key=rnd_id())
                    with col2:
                        st.button("Delete", on_click=self.delete, key=rnd_id())
                self.is_valid(show_warning=True)

    def set_editable(self, edit):
        self.edit = edit
        db_utils.save_crew(self)