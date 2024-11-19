from transformers import pipeline

# Load the pre-trained model for question-answering
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def answer_question(question, context):
    """
    Answer a question based on the provided legal text (context).
    """
    if not question or not context:
        return "Please provide both a question and context."
    
    # Use the QA model to get an answer
    result = qa_pipeline(question=question, context=context)
    return result["answer"]
