import argparse
import os
import cohere
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
from streamlit_chat import message


def run_chat_app(activeloop_dataset_path):
    """Run the chat application using the Streamlit framework."""
    st.title(f"{os.path.basename(activeloop_dataset_path)} GPT")

    co = cohere.Client(os.environ.get("COHERE_API_KEY"))

    db = DeepLake(
        dataset_path=activeloop_dataset_path,
        read_only=True,
    )


    if "generated" not in st.session_state:
        st.session_state["generated"] = ["i am ready to help you sir"]

    if "past" not in st.session_state:
        st.session_state["past"] = ["hello"]

    user_input = get_text()

    if user_input:
        output = search_db(co, db, user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i],
                    is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))


def get_text():
    """Create a Streamlit input field and return the user's input."""
    input_text = st.text_input("", key="input")
    return input_text


def search_db(co, db, query):
    """Search for a response to the query using Cohere's RetrievalQA."""
    context = db.retrieve(query)


    response = co.generate(
        model='command-xlarge-nightly', 
        prompt=f"Question: {query}\nContext: {context}\nAnswer:"
    )


    return response.generations[0].text.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activeloop_dataset_path", type=str, required=True)
    args = parser.parse_args()

    run_chat_app(args.activeloop_dataset_path)
