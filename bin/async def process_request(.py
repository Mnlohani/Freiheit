async def process_request(
   subtask_type: str = Form(...),
    image_resolution_type: str = Form(...),
    image: UploadFile = File(...),
    user_prompt: str = Form(...),
    language_of_response: str = Form(...)
):
    # Read image bytes
    image_bytes = await image.read()
    b64_image, _, _ = resize_and_getBase64(
        image_bytes, IMAGE_RESOLUTION[image_resolution_type]
    )

    response = get_response(
            llm, b64_image, subtask_type, user_prompt, language_of_response
        )
    return {"status": "success", "response":response}




 try:
            response = requests.post(
                    "http://localhost:9696/ask", 
                    data=payload, 
                    files=files,
                )
            
            if response.status_code == 200:
                result = response.json()
                ai_text = result.get("response", "No response field found")
                st.success(ai_text)
                # Display the response and convert it to speech
                st.write(ai_text)
                convert_text_to_speech(
                    result["response"],
                    LANGUAGE_OPTIONS[language_of_response],
                )
            else:
                st.error("This is on our side. Server Error!")
        
        except Exception as e:
            st.error(f"This is on our side. Server can not be connected: {e}")   
