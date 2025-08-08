from sclip_viewer.gradio import get_interface


demo = get_interface()
demo.launch(
    server_name="0.0.0.0", 
    server_port=7770, 
    share=False
)