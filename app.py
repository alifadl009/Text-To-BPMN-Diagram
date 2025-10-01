import streamlit as st
import os
import json
import graphviz
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration and Setup ---


# Configure Streamlit page
st.set_page_config(
    page_title="ProcessFlow AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean styling
st.markdown("""
    <style>
    /* Logo container */
    .logo-container {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
    }
    
    .logo-text {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        padding: 0;
    }
    
    .logo-subtext {
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    h1 {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize OpenAI client for DeepSeek API
try:
    client = OpenAI(
        api_key=st.secrets["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1"
    )
except Exception as e:
    st.error(f"Failed to initialize AI client. Is the DEEPSEEK_API_KEY set? Error: {e}")
    client = None

# --- Core Functions ---

def get_structured_process_flow(description: str):
    """
    Sends the process description to the DeepSeek LLM and asks for a structured
    BPMN-style diagram with swimlanes.
    """
    if not client:
        st.error("AI Client not initialized. Cannot generate flow.")
        return None

    system_prompt = """
    You are an expert BPMN 2.0 business process analyst. Your task is to analyze a given business process description
    and convert it into two formats:
    1. A structured JSON object suitable for rendering a BPMN diagram with swimlanes using Graphviz.
    2. A standard BPMN 2.0 XML string.

    The final output should be a single, valid JSON object with two top-level keys: "diagram_json" and "bpmn_xml".

    For "diagram_json":
    - It must have three keys: "swimlanes", "nodes", and "edges".
    - "swimlanes" is a list of objects, each with:
      - "id": A unique identifier (e.g., "customer", "agent", "system").
      - "label": The user-friendly name (e.g., "Customer", "Travel Agent", "System").
    - "nodes" is a list of objects, each with:
      - "id": A short, unique, snake_case identifier.
      - "label": The user-friendly text for the node.
      - "type": The BPMN element type ('start_event', 'end_event', 'task', 'gateway', 'message').
      - "swimlane": The swimlane id this node belongs to.
    - "edges" is a list of objects, each with:
      - "from": The source node "id".
      - "to": The destination node "id".
      - "label": An optional label for the edge (e.g., "Yes", "No", "Approved").

    Node type to shape mapping:
    - start_event: circle (small, green fill)
    - end_event: doublecircle (small, red fill)
    - task: box (rounded rectangle, blue fill)
    - gateway: diamond (yellow fill)
    - message: note (for messages/emails)

    For "bpmn_xml":
    - It must be a complete and valid BPMN 2.0 XML string with proper pool and lane definitions.

    Identify actors/roles as swimlanes. Create a clear, horizontal BPMN process flow with proper swimlanes.
    """

    user_prompt = f"Here is the business process description:\n\n---\n{description}\n---"

    try:
        with st.spinner("🤖 AI is analyzing your process and creating BPMN diagram..."):
            completion = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            response_content = completion.choices[0].message.content
            return json.loads(response_content)
    except json.JSONDecodeError as e:
        st.error(f"Error: The AI model returned an invalid JSON. Details: {e}")
        st.code(response_content, language="text")
        return None
    except Exception as e:
        st.error(f"An error occurred while communicating with the DeepSeek API: {e}")
        return None

def generate_bpmn_diagram(diagram_data: dict):
    """
    Generates a BPMN-style Graphviz diagram with swimlanes.
    """
    dot = graphviz.Digraph(comment='BPMN Process Flow')
    dot.attr('graph', rankdir='LR', splines='ortho', nodesep='0.8', ranksep='1.5', bgcolor='transparent')
    dot.attr('node', fontname='Arial', fontsize='11')
    dot.attr('edge', fontname='Arial', fontsize='9', color='#555555')

    # Create swimlanes as subgraphs (clusters)
    swimlanes = diagram_data.get("swimlanes", [])
    nodes = diagram_data.get("nodes", [])
    
    # Group nodes by swimlane
    swimlane_nodes = {sl["id"]: [] for sl in swimlanes}
    for node in nodes:
        swimlane_id = node.get("swimlane", "default")
        if swimlane_id in swimlane_nodes:
            swimlane_nodes[swimlane_id].append(node)
    
    # Create a cluster (swimlane) for each role/actor
    colors = ['#E3F2FD', '#F3E5F5', '#E8F5E9', '#FFF3E0', '#FCE4EC']
    for idx, swimlane in enumerate(swimlanes):
        with dot.subgraph(name=f'cluster_{swimlane["id"]}') as c:
            c.attr(label=swimlane["label"], style='filled', color='#666666', 
                   fillcolor=colors[idx % len(colors)], fontsize='14', fontname='Arial Bold')
            
            # Add nodes to this swimlane
            for node in swimlane_nodes.get(swimlane["id"], []):
                node_type = node.get("type", "task")
                
                if node_type == "start_event":
                    c.node(node["id"], label=node["label"], shape='circle', 
                           style='filled', fillcolor='#4CAF50', fontcolor='white', 
                           width='0.6', height='0.6', fixedsize='true')
                elif node_type == "end_event":
                    c.node(node["id"], label=node["label"], shape='doublecircle', 
                           style='filled', fillcolor='#F44336', fontcolor='white',
                           width='0.6', height='0.6', fixedsize='true')
                elif node_type == "task":
                    c.node(node["id"], label=node["label"], shape='box', 
                           style='rounded,filled', fillcolor='#2196F3', fontcolor='white')
                elif node_type == "gateway":
                    c.node(node["id"], label=node["label"], shape='diamond', 
                           style='filled', fillcolor='#FFC107', fontcolor='black')
                elif node_type == "message":
                    c.node(node["id"], label=node["label"], shape='note', 
                           style='filled', fillcolor='#FF9800', fontcolor='white')
                else:
                    c.node(node["id"], label=node["label"], shape='box', 
                           style='rounded,filled', fillcolor='#9E9E9E', fontcolor='white')

    # Add edges
    for edge in diagram_data.get("edges", []):
        dot.edge(
            tail_name=edge["from"],
            head_name=edge["to"],
            label=edge.get("label", "")
        )
    
    return dot


# --- Streamlit UI ---

def main():
    """The main function to run the Streamlit application."""
    
    # Logo
    st.markdown("""
        <div class="logo-container">
            <div class="logo-text">🤖 ProcessFlow AI</div>
            <div class="logo-subtext">Transform Text into Professional BPMN Diagrams</div>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "diagram_generated" not in st.session_state:
        st.session_state.diagram_generated = False
        st.session_state.diagram_data = None
        st.session_state.bpmn_xml = None
        st.session_state.graph_obj = None

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### 📊 Export Options")
        
        if st.session_state.diagram_generated:
            st.download_button(
                label="⚙️ Download BPMN XML",
                data=st.session_state.bpmn_xml,
                file_name="process_flow.bpmn",
                mime="application/xml",
                use_container_width=True,
            )
            st.success("✅ Diagram generated!")
        else:
            st.info("Generate a diagram first to enable export.")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **ProcessFlow AI** uses advanced AI to convert your business process descriptions 
        into professional BPMN diagrams with:
        - **Swimlanes** for different roles
        - **Tasks** and activities
        - **Events** (start/end)
        - **Gateways** for decisions
        - **Message flows**
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Mention different roles/actors
        - Describe decision points clearly
        - Include start and end states
        - Mention any messages or notifications
        """)

    # --- Main Area ---
    st.markdown("## 📝 Describe Your Business Process")
    
    user_input = st.text_area(
        "Enter your process description below:",
        height=200,
        placeholder="""Example: A customer requests a cab booking through the app. The travel agent receives the request and checks availability. If available, the agent proposes booking status to the customer. If not available, the agent gets an alternative time. The customer reviews and either accepts or rejects. If accepted, the agent confirms the booking and notifies the cab driver. The cab driver picks up the customer.""",
        key="process_description"
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        generate_button = st.button("✨ Generate BPMN Diagram", type="primary", use_container_width=True, disabled=(not user_input.strip()))

    if generate_button:
        if len(user_input.strip()) < 20:
            st.warning("⚠️ Please provide a more detailed process description (at least 20 characters).")
        else:
            structured_output = get_structured_process_flow(user_input)
            if structured_output and "diagram_json" in structured_output and "bpmn_xml" in structured_output:
                st.session_state.diagram_data = structured_output["diagram_json"]
                st.session_state.bpmn_xml = structured_output["bpmn_xml"]
                st.session_state.graph_obj = generate_bpmn_diagram(st.session_state.diagram_data)
                st.session_state.diagram_generated = True
                st.success("✅ BPMN Diagram generated successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to generate diagram. Please try rephrasing your input.")
                st.session_state.diagram_generated = False

    # --- Display Diagram ---
    if st.session_state.diagram_generated:
        st.markdown("---")
        st.markdown("## 🎨 Your BPMN Process Diagram")
        st.graphviz_chart(st.session_state.graph_obj.source, use_container_width=True)

        # Expander to show the raw data
        with st.expander("🔍 View Raw AI Output"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Diagram JSON Data")
                st.json(st.session_state.diagram_data)
            with col2:
                st.markdown("#### BPMN 2.0 XML")
                st.code(st.session_state.bpmn_xml, language="xml")

if __name__ == "__main__":

    main()
