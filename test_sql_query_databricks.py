
# Chat input for user query
"""if prompt := st.chat_input("Ask a question or request data:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat UI
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query Databricks based on user input
    try:
        with dbsql.connect(
            server_hostname=DATABRICKS_SERVER_HOSTNAME,
            http_path=HTTP_PATH,
            access_token=ACCESS_TOKEN
        ) as connection:
            
            # Example query based on the user's input
            query = f"SELECT * FROM your_table WHERE column LIKE '%{prompt}%' LIMIT 5"
            with connection.cursor() as cursor:
                cursor.execute(query)
                data = cursor.fetchall()

            # Format and display the Databricks results
            if data:
                result_message = "Here are the results from Databricks:"
                for row in data:
                    result_message += f"\n{row}"
            else:
                result_message = "No results found for your query."

            # Add Databricks result to chat history
            st.session_state.messages.append({"role": "assistant", "content": result_message})
            with st.chat_message("assistant"):
                st.markdown(result_message)

    except Exception as e:
        error_message = f"An error occurred while querying Databricks: {e}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        with st.chat_message("assistant"):
            st.markdown(error_message)

    # Save the current chat session to history
    st.session_state.chat_history.append(st.session_state.messages.copy())

    # Clear the current chat after sending response
    st.session_state.messages = []  # Reset for the next interaction

"""