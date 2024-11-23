import gradio as gr
import requests
import io
from PIL import Image

# Define a function to send the image to an API and handle the response
def send_to_api(image):
    try:
        # Convert the Gradio image input to a PIL image
        pil_image = Image.fromarray(image)
        
        # Convert the PIL image to bytes for sending
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()
        
        # API endpoint
        api_url = "http://10.127.30.125:8000/inference/"
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        
        # Send the request
        response = requests.post(api_url, files=files)
        print(f"Status Code: {response.status_code}")  # Debug: Print status code
        print(f"Response: {response.text}")  # Debug: Print response text
        
        response.raise_for_status()  # Raise error if the request failed
        
        # Parse the JSON response
        response_data = response.json()
        
        # Display the prediction
        if response_data.get("status") == "success":
            class_ = response_data.get("class", "Unknown")
            prediction = response_data.get("prediction", "Unknown")
            return f"Predicted Class: {class_}, Grip: {prediction}"
        else:
            return f"Error: {response_data.get('detail', 'Unknown error')}"
    
    except requests.exceptions.RequestException as e:
        return f"Error communicating with API: {e}"
    except Exception as e:
        return f"An error occurred: {e}"


# Create the Gradio interface
iface = gr.Interface(
    fn=send_to_api,
    inputs=gr.Image(type="numpy"),  # Accepts an image as input
    outputs="text",  # Displays the API response as text
    title="Image to API",
    description="Upload an image and send it to an external API for processing. The response will be displayed here."
)

# Launch the app
if __name__ == "__main__":
    # iface.launch()
    iface.launch(server_name="0.0.0.0", server_port=7860)

