import streamlit as st
import requests
from PIL import Image
import io
import openai
from openai import AzureOpenAI
import base64
import httpx

# Setting up the HTTP client
http_client = httpx.Client(verify=False)

# Setting up Important variables
openai.api_base = "**"
openai.api_type = "azure"
openai.api_version = "2024-02-01"
openai.api_key = "***"  # Use a secure way to store your API key
model = "gpt-4o"

# Setting Up AzureOpenAI Client
client = AzureOpenAI(
    azure_endpoint=openai.api_base,
    api_key=openai.api_key,
    api_version=openai.api_version,
    http_client=http_client
)

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to determine if the text is likely tabular
def is_tabular(text):
    lines = text.split('\n')
    if len(lines) < 2:
        return False
    delimiter_counts = [line.count('\t') + line.count(',') for line in lines if line.strip()]
    if delimiter_counts:
        # Check if there's a consistent number of delimiters per line
        first_count = delimiter_counts[0]
        return all(count == first_count for count in delimiter_counts)
    return False

# Function to recognize handwritten text using GPT-4o model
def recognize_handwritten_text(image_bytes):
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What’s in this image? Please write the same text as in the image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/webp;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=3000  # Adjusted max tokens
        )
        if response and response.choices:
            return response.choices[0].message.content
        else:
            st.error("No response or empty choices from the API.")
            return None
    except Exception as e:
        st.error(f"An error occurred while calling the Azure OpenAI API: {e}")
        return None

# Streamlit interface
st.title("Handwritten Document Recognition and Data Digitization")

uploaded_file = st.file_uploader("Upload a handwritten document", type=["png", "jpg", "jpeg","jfif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Handwritten Document", use_column_width=True)
    
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    if st.button("Recognize Text"):
        with st.spinner("Recognizing text..."):
            extracted_text = recognize_handwritten_text(img_byte_arr)
            if extracted_text:
                st.text_area("Extracted Text", extracted_text, height=300)
                
                # Check if the text is tabular and display as structured table
                if is_tabular(extracted_text):
                    st.write("### Structured Table")
                    rows = extracted_text.split('\n')
                    table_data = [row.split() for row in rows if row.strip()]
                    st.table(table_data)
                else:
                    st.write("Text does not appear to be in tabular format.")

                # Generate summary of the text
                st.write("### Summary")
                try:
                    summary_response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": f"Provide a summary for the following text: {extracted_text}"
                            }
                        ],
                        max_tokens=150
                    )
                    if summary_response and summary_response.choices:
                        summary = summary_response.choices[0].message.content
                        st.write(summary)
                    else:
                        st.error("No response or empty choices from the API for summary.")
                except Exception as e:
                    st.error(f"An error occurred while generating the summary: {e}")

                # Provide download option
                st.write("### Download Extracted Text")
                st.download_button(
                    label="Download as Text File",
                    data=extracted_text,
                    file_name="extracted_text.txt",
                    mime="text/plain"
                )
