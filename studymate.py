import streamlit as st
import PyPDF2
import re
from typing import List, Tuple
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class StudyMate:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
            return text.strip()
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better analysis."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into smaller chunks for better processing."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk + sentence) <= chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def find_relevant_chunks(self, question: str, text_chunks: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """Find most relevant text chunks for the given question."""
        if not text_chunks:
            return []
        all_texts = [question] + text_chunks
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            question_vector = tfidf_matrix[0]
            chunk_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(question_vector, chunk_vectors).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            relevant_chunks = [(text_chunks[i], similarities[i]) for i in top_indices if similarities[i] > 0.1]
            return relevant_chunks
        except Exception as e:
            st.error(f"Error in finding relevant chunks: {str(e)}")
            return []

    def generate_answer(self, question: str, relevant_chunks: List[Tuple[str, float]]) -> str:
        """Generate answer based on relevant text chunks."""
        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question. Please try rephrasing or asking about different topics."

        question_lower = question.lower()
        if any(k in question_lower for k in ['summary', 'summarize', 'overview']):
            return self._generate_summary(relevant_chunks)
        elif any(k in question_lower for k in ['main point', 'key point', 'important', 'main idea']):
            return self._generate_key_points(relevant_chunks)
        elif any(k in question_lower for k in ['define', 'definition', 'what is', 'what are']):
            return self._generate_definition(relevant_chunks)
        elif any(k in question_lower for k in ['how', 'method', 'process', 'procedure']):
            return self._generate_process_answer(relevant_chunks)
        elif any(k in question_lower for k in ['why', 'reason', 'because']):
            return self._generate_reasoning_answer(relevant_chunks)
        else:
            return self._generate_general_answer(relevant_chunks)

    # Helper methods for answer generation
    def _generate_summary(self, chunks: List[Tuple[str, float]]) -> str:
        summary_parts = [f"• {chunk[:200]}..." if len(chunk) > 200 else f"• {chunk}" for chunk, _ in chunks[:3]]
        return "📋 **Summary:**\n\n" + "\n\n".join(summary_parts)

    def _generate_key_points(self, chunks: List[Tuple[str, float]]) -> str:
        points = []
        for i, (chunk, _) in enumerate(chunks[:3]):
            first_sentence = chunk.split('.')[0] + '.'
            if len(first_sentence) > 150:
                first_sentence = chunk[:150] + "..."
            points.append(f"{i+1}. {first_sentence}")
        return "🎯 **Key Points:**\n\n" + "\n\n".join(points)

    def _generate_definition(self, chunks: List[Tuple[str, float]]) -> str:
        best_chunk = chunks[0][0] if chunks else ""
        return f"📖 **Definition/Explanation:**\n\n{best_chunk[:300]}..." if len(best_chunk) > 300 else f"📖 **Definition/Explanation:**\n\n{best_chunk}"

    def _generate_process_answer(self, chunks: List[Tuple[str, float]]) -> str:
        process_info = chunks[0][0] if chunks else ""
        return f"⚙️ **Process/Method:**\n\n{process_info[:400]}..." if len(process_info) > 400 else f"⚙️ **Process/Method:**\n\n{process_info}"

    def _generate_reasoning_answer(self, chunks: List[Tuple[str, float]]) -> str:
        reasoning = chunks[0][0] if chunks else ""
        return f"🤔 **Reasoning/Explanation:**\n\n{reasoning[:350]}..." if len(reasoning) > 350 else f"🤔 **Reasoning/Explanation:**\n\n{reasoning}"

    def _generate_general_answer(self, chunks: List[Tuple[str, float]]) -> str:
        best_chunk = chunks[0][0] if chunks else ""
        confidence = chunks[0][1] if chunks else 0
        confidence_emoji = "🎯" if confidence > 0.5 else "💡" if confidence > 0.3 else "🔍"
        answer = f"{confidence_emoji} **Answer:**\n\n{best_chunk}"
        if len(chunks) > 1 and chunks[1][1] > 0.2:
            additional_info = chunks[1][0][:200] + "..." if len(chunks[1][0]) > 200 else chunks[1][0]
            answer += f"\n\n📌 **Additional Information:**\n{additional_info}"
        return answer


def main():
    st.set_page_config(page_title="StudyMate - AI PDF Q&A", page_icon="📚", layout="wide")

    # CSS
    st.markdown("""
    <style>
    .main-header {background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;}
    .main-header h1 {color: white; text-align: center; margin-bottom: 0.5rem;}
    .main-header p {color: white; text-align: center; opacity: 0.9; font-size: 1.1rem;}
    .question-box {background: #e3f2fd; padding: 1rem; border-radius: 8px; border-left: 4px solid #2196f3; margin: 1rem 0;}
    .answer-box {background: #f3e5f5; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #9c27b0; margin: 1rem 0;}
    .stats-box {background: #e8f5e8; padding: 1rem; border-radius: 8px; text-align: center;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header"><h1>📚 StudyMate</h1><p>AI-Powered PDF-Based Q&A System for Students</p></div>', unsafe_allow_html=True)

    # Initialize session state
    if 'studymate' not in st.session_state:
        st.session_state.studymate = StudyMate()
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""
    if 'text_chunks' not in st.session_state:
        st.session_state.text_chunks = []
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []

    # Sidebar
    with st.sidebar:
        st.header("📋 Document Info")
        if st.session_state.pdf_text:
            st.markdown(f"""
            <div class="stats-box">
                <h4>📊 Statistics</h4>
                <p><strong>Words:</strong> {len(st.session_state.pdf_text.split()):,}</p>
                <p><strong>Chunks:</strong> {len(st.session_state.text_chunks)}</p>
                <p><strong>Status:</strong> ✅ Ready</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Upload a PDF to get started!")

        st.header("💡 Sample Questions")
        sample_questions = [
            "What is the main topic of this document?",
            "Summarize the key points",
            "What are the important definitions?",
            "How does the process work?",
            "What are the main conclusions?",
            "Why is this topic important?"
        ]
        for question_text in sample_questions:
            if st.button(f"💭 {question_text}", key=f"sample_{hash(question_text)}"):
                st.session_state.ask_question_area = question_text  # <- set the text area value directly

    # Main content
    col1, col2 = st.columns([1, 1])

    # PDF Upload
    with col1:
        st.header("📄 Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
        if uploaded_file is not None:
            if st.button("🔄 Process PDF", type="primary", key="process_pdf"):
                with st.spinner("Extracting text..."):
                    pdf_text = st.session_state.studymate.extract_text_from_pdf(uploaded_file)
                    if pdf_text:
                        processed_text = st.session_state.studymate.preprocess_text(pdf_text)
                        chunks = st.session_state.studymate.chunk_text(processed_text)
                        st.session_state.pdf_text = processed_text
                        st.session_state.text_chunks = chunks
                        st.success(f"✅ PDF processed! {len(chunks)} chunks created.")

        if st.session_state.pdf_text:
            st.subheader("📖 Document Preview")
            preview_text = st.session_state.pdf_text[:500] + "..." if len(st.session_state.pdf_text) > 500 else st.session_state.pdf_text
            st.text_area("First 500 characters:", preview_text, height=150, disabled=True, key="preview_area")

    # Ask Questions
    with col2:
        st.header("💭 Ask Questions")
        current_question = st.session_state.get('current_question', '')
        question = st.text_area(
            "Enter your question:",
            value=st.session_state.get("ask_question_area", ""),
            height=100,
            placeholder="Ask anything about your PDF content...",
            disabled=not bool(st.session_state.pdf_text),
            key="ask_question_area"
        )

        if 'current_question' in st.session_state:
            del st.session_state.current_question

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            ask_button = st.button("🤖 Ask StudyMate", type="primary",
                                   disabled=not bool(st.session_state.pdf_text and question.strip()), key="ask_button")
        with col_btn2:
            if st.button("🗑️ Clear History", key="clear_history_button"):
                st.session_state.qa_history = []
                st.success("History cleared!")

        # Process question
        if ask_button and question.strip() and st.session_state.pdf_text:
            with st.spinner("🔍 Analyzing document and generating answer..."):
                relevant_chunks = st.session_state.studymate.find_relevant_chunks(question, st.session_state.text_chunks)
                answer = st.session_state.studymate.generate_answer(question, relevant_chunks)
                st.session_state.qa_history.insert(0, {
                    'question': question,
                    'answer': answer,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'relevance_score': relevant_chunks[0][1] if relevant_chunks else 0
                })

    # Display Q&A History
    if st.session_state.qa_history:
        st.header("📝 Q&A History")
        for i, qa in enumerate(st.session_state.qa_history):
            with st.expander(f"❓ {qa['question'][:100]}..." if len(qa['question']) > 100 else f"❓ {qa['question']}", expanded=(i==0)):
                st.markdown(f'<div class="question-box"><strong>Question:</strong> {qa["question"]}<br><small>⏰ {qa["timestamp"]} | 📊 Relevance: {qa["relevance_score"]:.2f}</small></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-box">{qa["answer"]}</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #666; padding: 1rem;"><p>🚀 StudyMate v1.0 | Built with ❤️ for Students</p></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
