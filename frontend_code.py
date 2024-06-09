import streamlit as st
import backend_code as bc
import store1 as s1

def main():
    st.header("Welcome to Personalized Assistant using AWS Bedrock")

    # Display chat history
    

    # Text area for user input (at the bottom)
    query = st.text_area("Ask a Question from the Retirement Services")
    subbtn=st.button("Submit")
    clrbtn=st.button("Clear Chat")
    if subbtn:
        with st.spinner("Thinking..."):
            res, lnk = bc.lambda_handler(query, 'context')
            # Update chat history
            s1.chat_history[query] = res+'\n'+lnk
            st.subheader("Chat History:")
            for sender, message in s1.chat_history.items():
                st.write('<b>You: </b>'+sender,unsafe_allow_html=True)
                result=message.splitlines()
                st.write(f"<b style='color:Green'>Bot: </b><span style='color:Green'>{result[0]}</span>",unsafe_allow_html=True)
                st.write(f"<span style='color:Black;font-weight: bold;'>{result[1]}</span>",unsafe_allow_html=True)

    if clrbtn:
        s1.chat_history.clear()

if __name__ == "__main__":
    main()
