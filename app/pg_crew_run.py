import re
import streamlit as st
from streamlit import session_state as ss
import threading
import ctypes
import queue
import json
import time
import traceback
import os
from console_capture import ConsoleCapture
from pyvis.network import Network
import streamlit.components.v1 as components

class PageCrewRun:
    def __init__(self):
        self.name = "Kickoff!"
        self.maintain_session_state()
    
    @staticmethod
    def maintain_session_state():
        defaults = {
            'crew_thread': None,
            'result': None,
            'running': False,
            'message_queue': queue.Queue(),
            'selected_crew_name': None,
            'placeholders': {},
            'console_output': [],
            'last_update': time.time(),
            'console_expanded': ss.get('running', False),
        }
        for key, value in defaults.items():
            if key not in ss:
                ss[key] = value

    def display_graph(self):
        # Create a PyVis network
        net = Network(height="600px", width="100%", directed=True)

        # Retrieve the crews from session state
        crews = ss.get("crews", [])

        # Add nodes and edges for each crew and its relationships
        for crew in crews:
            # Add a node for the crew
            agent_names = "\n".join(agent.role for agent in crew.agents)  # Assuming each crew has an 'agents' attribute
            crew_label = f"{crew.name}\nAgents:\n{agent_names}"
            
            net.add_node(
                crew.name,
                label=crew_label,
                title=crew.name,
                shape="box"  # Change node shape to a square (box)
            )

            # Add edges for relationships if they exist
            if hasattr(crew, "related_crews"):
                for related_crew in crew.related_crews:
                    net.add_edge(crew.name, related_crew.name)

        # Set basic interaction options
        net.set_options(json.dumps({
            "physics": {"enabled": False},
            "interaction": {
                "dragNodes": True,
                "dragView": True,
                "selectable": True,
                "zoomView": True,
                "multiselect": False
            },
            "manipulation": {
                "enabled": True
            }
        }))
        net.write_html("custom_graph_rendered.html")

        # Render the graph in Streamlit
        components.html(open("custom_graph_rendered.html", "r").read(), height=600)

    
    # st.title("Crew Visualization")
    # display_graph()
                
    @staticmethod
    def extract_placeholders(text):
        return re.findall(r'\{(.*?)\}', text)

    def get_placeholders_from_crew(self, crew):
        placeholders = set()
        attributes = ['description', 'expected_output', 'role', 'backstory', 'goal']
        
        for task in crew.tasks:
            placeholders.update(self.extract_placeholders(task.description))
            placeholders.update(self.extract_placeholders(task.expected_output))
        
        for agent in crew.agents:
            for attr in attributes[2:]:
                placeholders.update(self.extract_placeholders(getattr(agent, attr)))
        
        return placeholders

    def run_crew(self, crewai_crew, inputs, message_queue):
        if (str(os.getenv('AGENTOPS_ENABLED')).lower() in ['true', '1']) and not ss.get('agentops_failed', False):
            import agentops
            agentops.start_session()
        try:
            result = crewai_crew.kickoff(inputs=inputs)
            message_queue.put({"result": result})
        except Exception as e:
            if (str(os.getenv('AGENTOPS_ENABLED')).lower() in ['true', '1']) and not ss.get('agentops_failed', False):                       
                agentops.end_session()
            stack_trace = traceback.format_exc()
            print(f"Error running crew: {str(e)}\n{stack_trace}")
            message_queue.put({"result": f"Error running crew: {str(e)}", "stack_trace": stack_trace})
        finally:
            if hasattr(ss, 'console_capture'):
                ss.console_capture.stop()

    def get_mycrew_by_name(self, crewname):
        return next((crew for crew in ss.crews if crew.name == crewname), None)

    def draw_placeholders(self, crew):
        placeholders = self.get_placeholders_from_crew(crew)
        if placeholders:
            st.write('Placeholders to fill in:')
            for placeholder in placeholders:
                placeholder_key = f'placeholder_{placeholder}'
                ss.placeholders[placeholder_key] = st.text_input(
                    label=placeholder,
                    key=placeholder_key,
                    value=ss.placeholders.get(placeholder_key, ''),
                    disabled=ss.running
                )

    def draw_crews(self):
        if 'crews' not in ss or not ss.crews:
            st.write("No crews defined yet.")
            ss.selected_crew_name = None  # Reset selected crew name if there are no crews
            return

        # Check if the selected crew name still exists
        if ss.selected_crew_name not in [crew.name for crew in ss.crews]:
            ss.selected_crew_name = None

        selected_crew_name = st.selectbox(
            label="Select crew to run",
            options=[crew.name for crew in ss.crews],
            index=0 if ss.selected_crew_name is None else [crew.name for crew in ss.crews].index(ss.selected_crew_name) if ss.selected_crew_name in [crew.name for crew in ss.crews] else 0,
            disabled=ss.running
        )

        if selected_crew_name != ss.selected_crew_name:
            ss.selected_crew_name = selected_crew_name
            st.rerun()

        selected_crew = self.get_mycrew_by_name(ss.selected_crew_name)

        if selected_crew:
            selected_crew.draw(expanded=False,buttons=False)
            self.draw_placeholders(selected_crew)
            
            if not selected_crew.is_valid(show_warning=True):
                st.error("Selected crew is not valid. Please fix the issues.")
            self.control_buttons(selected_crew)

    def control_buttons(self, selected_crew):
        if st.button('Run crew!', disabled=not selected_crew.is_valid() or ss.running):
            inputs = {key.split('_')[1]: value for key, value in ss.placeholders.items()}
            ss.result = None
            
            try:
                crew = selected_crew.get_crewai_crew(full_output=True)
            except Exception as e:
                st.exception(e)
                traceback.print_exc()
                return

            ss.console_capture = ConsoleCapture()
            ss.console_capture.start()
            ss.console_output = []  # Reset výstupu

            ss.running = True
            ss.crew_thread = threading.Thread(
                target=self.run_crew,
                kwargs={
                    "crewai_crew": crew,
                    "inputs": inputs,
                    "message_queue": ss.message_queue
                }
            )
            ss.crew_thread.start()
            ss.result = None
            ss.running = True            
            st.rerun()

        if st.button('Stop crew!', disabled=not ss.running):
            self.force_stop_thread(ss.crew_thread)
            if hasattr(ss, 'console_capture'):
                ss.console_capture.stop()
            ss.message_queue.queue.clear()
            ss.running = False
            ss.crew_thread = None
            ss.result = None
            st.success("Crew stopped successfully.")
            st.rerun()

    def display_result(self):
        # Ensure the console output expander is expanded when the kickoff starts
        if ss.running:
            ss.console_expanded = True  # Automatically expand the console if running
        elif "console_expanded" not in ss:
            ss.console_expanded = False  # Default to collapsed when not running

        if ss.running and ss.page != "Kickoff!":
            ss.page = "Kickoff!"
            st.rerun()

        console_container = st.empty()
        
        with console_container.container():
            with st.expander("Console Output", expanded=ss.console_expanded):
                col1, col2 = st.columns([6,1])
                with col2:
                    if st.button("Clear console"):
                        ss.console_output = []
                        st.rerun()

                console_text = "\n".join(ss.console_output)
                st.code(console_text, language=None)

        if ss.result is not None:
            if isinstance(ss.result, dict):
                if 'final_output' in ss.result["result"]:
                    st.expander("Final output", expanded=True).write(ss.result["result"]['final_output'])
                elif hasattr(ss.result["result"], 'raw'):
                    st.expander("Final output", expanded=True).write(ss.result['result'].raw)
                st.expander("Full output", expanded=False).write(ss.result)
            else:
                st.error(ss.result)
        elif ss.running and ss.crew_thread is not None:
            console_container = st.empty()  # Placeholder for console or spinner
            with console_container.container():
                with st.spinner("Running crew..."):  # Spinner persists until running ends
                    while ss.running:
                        if hasattr(ss, 'console_capture'):
                            new_output = ss.console_capture.get_output()
                            if new_output:
                                ss.console_output.extend(new_output)

                        try:
                            # Check for messages from the queue
                            message = ss.message_queue.get_nowait()
                            ss.result = message
                            ss.running = False
                            ss.console_expanded = False
                            if hasattr(ss, 'console_capture'):
                                ss.console_capture.stop()
                            break  # Exit the loop when the crew finishes
                        except queue.Empty:
                            time.sleep(0.5)  # Small delay to avoid excessive CPU usage

                    st.rerun()  # Reload the page to update UI state


    @staticmethod
    def force_stop_thread(thread):
        if thread:
            tid = ctypes.c_long(thread.ident)
            if tid:
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(SystemExit))
                if res == 0:
                    st.error("Nonexistent thread id")
                else:
                    st.success("Thread stopped successfully.")

    def draw(self):
        st.subheader(self.name)
        self.draw_crews()
        self.display_result()