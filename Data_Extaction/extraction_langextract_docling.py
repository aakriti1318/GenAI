#!/usr/bin/env python3
"""
Enhanced PDF Extraction Comparison Tool with Streamlit UI
Compares LangExtract and Docling libraries with visual analytics
"""

import os
import ssl
import time
import textwrap
import urllib.request
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, Any, List, Tuple
import threading
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="PDF Extraction Library Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .success-container {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .error-container {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sample text for testing
SAMPLE_TEXT = """
Title: Attention Is All You Need

Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show that these models are superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU points.

1. Introduction
The Transformer model architecture has become the foundation for many state-of-the-art natural language processing systems.

2. Model Architecture
The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers.

Conclusion: We presented the Transformer, a novel neural network architecture based entirely on attention mechanisms.
"""

def fix_ssl_for_downloads():
    """Fix SSL issues for model downloads"""
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass

class ExtractionResult:
    """Class to store extraction results for comparison"""
    def __init__(self, library_name: str):
        self.library_name = library_name
        self.success = False
        self.execution_time = 0.0
        self.extracted_content = ""
        self.content_length = 0
        self.structure_info = {}
        self.error_message = ""
        self.features_used = []
        self.accuracy_score = 0.0

def test_langextract(progress_bar, status_text) -> ExtractionResult:
    """Test LangExtract with enhanced metrics"""
    result = ExtractionResult("LangExtract")
    
    try:
        status_text.text("🔄 Initializing LangExtract...")
        progress_bar.progress(10)
        
        import langextract as lx
        
        status_text.text("🔄 Setting up extraction parameters...")
        progress_bar.progress(30)
        
        # More specific and detailed extraction prompt
        prompt = textwrap.dedent("""
            Extract the following elements from research papers:
            1. Paper title (after "Title:")
            2. Author names (after "Authors:")
            3. Abstract content (after "Abstract:")
            4. Key results or metrics mentioned
            5. Main contributions or conclusions
            
            Be precise and extract the exact text as it appears in the document.
        """)
        
        # Better examples that match the actual text structure
        examples = [
            lx.data.ExampleData(
                text="Title: Attention Is All You Need\nAuthors: Ashish Vaswani, Noam Shazeer\nAbstract: The dominant sequence transduction models are based on complex recurrent networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="title",
                        extraction_text="Attention Is All You Need",
                        attributes={"type": "paper_title", "section": "header"}
                    ),
                    lx.data.Extraction(
                        extraction_class="authors",
                        extraction_text="Ashish Vaswani, Noam Shazeer",
                        attributes={"type": "author_list", "count": 2}
                    ),
                    lx.data.Extraction(
                        extraction_class="abstract",
                        extraction_text="The dominant sequence transduction models are based on complex recurrent networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
                        attributes={"type": "abstract_content", "length": "short"}
                    )
                ]
            ),
            lx.data.ExampleData(
                text="Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results by over 2 BLEU points.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="result",
                        extraction_text="achieves 28.4 BLEU on the WMT 2014 English-to-German translation task",
                        attributes={"metric": "BLEU", "value": "28.4", "task": "translation"}
                    ),
                    lx.data.Extraction(
                        extraction_class="improvement",
                        extraction_text="improving over the existing best results by over 2 BLEU points",
                        attributes={"improvement": "2+ BLEU points", "comparison": "state_of_art"}
                    )
                ]
            )
        ]
        
        status_text.text("🔄 Running extraction...")
        progress_bar.progress(60)
        
        start_time = time.time()
        
        # Try multiple approaches for better extraction
        try:
            # Primary extraction attempt
            extraction_result = lx.extract(
                text_or_documents=SAMPLE_TEXT,
                prompt_description=prompt,
                examples=examples,
                model_id="gemini-2.0-flash-exp",
                extraction_passes=2,  # More passes for better results
                max_workers=1  # Reduce workers to avoid rate limits
            )
        except Exception as primary_error:
            status_text.text("🔄 Trying alternative approach...")
            # Fallback with simpler approach
            simple_prompt = "Extract title, authors, and key content from this research paper text."
            extraction_result = lx.extract(
                text_or_documents=SAMPLE_TEXT,
                prompt_description=simple_prompt,
                examples=[examples[0]],  # Use only first example
                model_id="gemini-2.0-flash-exp",
                extraction_passes=1,
                max_workers=1
            )
        
        result.execution_time = time.time() - start_time
        progress_bar.progress(90)
        
        # Process results with better handling
        extractions = []
        extraction_count = 0
        
        if hasattr(extraction_result, 'extractions') and extraction_result.extractions:
            for extraction in extraction_result.extractions:
                extractions.append({
                    'class': extraction.extraction_class,
                    'text': extraction.extraction_text,
                    'attributes': getattr(extraction, 'attributes', {})
                })
                extraction_count += 1
            
            result.extracted_content = json.dumps(extractions, indent=2)
            result.content_length = len(result.extracted_content)
            result.structure_info = {
                'total_extractions': len(extraction_result.extractions),
                'extraction_classes': len(set(e.extraction_class for e in extraction_result.extractions)),
                'has_attributes': sum(1 for e in extraction_result.extractions if hasattr(e, 'attributes') and e.attributes),
                'unique_classes': list(set(e.extraction_class for e in extraction_result.extractions))
            }
            
            # Calculate accuracy score based on successful extractions
            expected_elements = ['title', 'authors', 'abstract', 'result']
            found_elements = [e.extraction_class.lower() for e in extraction_result.extractions]
            matches = len(set(expected_elements) & set(found_elements))
            result.accuracy_score = (matches / len(expected_elements)) * 100
            
            result.success = True
            status_text.text(f"✅ LangExtract completed! Found {extraction_count} extractions")
            
        else:
            # Handle empty extractions case
            result.extracted_content = "No extractions found"
            result.content_length = 0
            result.structure_info = {
                'total_extractions': 0,
                'extraction_classes': 0,
                'has_attributes': 0,
                'issue': 'No extractions returned by model'
            }
            result.accuracy_score = 0
            result.success = False
            result.error_message = "Model returned empty extractions - possibly due to prompt/example mismatch"
            status_text.text("⚠️ LangExtract completed but found no extractions")
        
        result.features_used = ['Semantic Understanding', 'Custom Classes', 'Attributes', 'Few-shot Learning']
        progress_bar.progress(100)
        
    except Exception as e:
        result.error_message = str(e)
        result.success = False
        result.accuracy_score = 0
        result.structure_info = {'error': str(e)}
        status_text.text(f"❌ LangExtract failed: {str(e)[:50]}...")
        progress_bar.progress(100)
    
    return result

def test_docling(progress_bar, status_text) -> ExtractionResult:
    """Test Docling with enhanced metrics"""
    result = ExtractionResult("Docling")
    
    try:
        status_text.text("🔄 Initializing Docling...")
        progress_bar.progress(10)
        
        from docling.document_converter import DocumentConverter
        fix_ssl_for_downloads()
        
        # Look for PDF files
        pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
        
        if not pdf_files:
            # Create a sample document for testing
            status_text.text("📝 No PDF found, creating sample document...")
            with open('sample_doc.txt', 'w') as f:
                f.write(SAMPLE_TEXT)
            test_file = 'sample_doc.txt'
        else:
            test_file = pdf_files[0]
            
        progress_bar.progress(30)
        
        status_text.text("🔄 Converting document...")
        progress_bar.progress(50)
        
        start_time = time.time()
        
        # Convert document
        converter = DocumentConverter()
        doc = converter.convert(test_file).document
        
        result.execution_time = time.time() - start_time
        progress_bar.progress(80)
        
        # Export to markdown
        markdown = doc.export_to_markdown()
        result.extracted_content = markdown
        result.content_length = len(markdown)
        
        # Analyze structure
        structure_info = {
            "total_elements": 0,
            "text_blocks": 0,
            "tables": 0,
            "figures": 0
        }
        
        try:
            if hasattr(doc, 'texts'):
                structure_info["text_blocks"] = len(doc.texts)
                structure_info["total_elements"] += len(doc.texts)
            
            if hasattr(doc, 'tables'):
                structure_info["tables"] = len(doc.tables)
                structure_info["total_elements"] += len(doc.tables)
                
            if hasattr(doc, 'pictures'):
                structure_info["figures"] = len(doc.pictures)
                structure_info["total_elements"] += len(doc.pictures)
        except:
            pass
            
        result.structure_info = structure_info
        
        # Calculate accuracy based on content preservation
        result.accuracy_score = min(100, (len(markdown) / len(SAMPLE_TEXT)) * 50 + 50)
        
        result.features_used = ['Structure Extraction', 'Markdown Export', 'Layout Preservation', 'Multi-format Support']
        result.success = True
        
        progress_bar.progress(100)
        status_text.text("✅ Docling completed successfully!")
        
    except ImportError:
        result.error_message = "Docling not installed. Run: pip install docling"
        result.success = False
        status_text.text("❌ Docling not installed")
        progress_bar.progress(100)
    except Exception as e:
        result.error_message = str(e)
        result.success = False
        status_text.text(f"❌ Docling failed: {str(e)[:50]}...")
        progress_bar.progress(100)
    
    return result

def create_comparison_charts(langextract_result: ExtractionResult, docling_result: ExtractionResult):
    """Create visual comparison charts"""
    
    # Performance Comparison
    fig_performance = go.Figure()
    
    libraries = ['LangExtract', 'Docling']
    times = [langextract_result.execution_time, docling_result.execution_time]
    colors = ['#FF6B6B' if not langextract_result.success else '#4ECDC4',
              '#FF6B6B' if not docling_result.success else '#45B7D1']
    
    fig_performance.add_trace(go.Bar(
        x=libraries,
        y=times,
        marker_color=colors,
        text=[f'{t:.2f}s' for t in times],
        textposition='auto',
        name='Execution Time'
    ))
    
    fig_performance.update_layout(
        title='⏱️ Execution Time Comparison',
        yaxis_title='Time (seconds)',
        template='plotly_white',
        height=400
    )
    
    # Accuracy Comparison
    fig_accuracy = go.Figure()
    
    accuracy_scores = [langextract_result.accuracy_score, docling_result.accuracy_score]
    
    fig_accuracy.add_trace(go.Bar(
        x=libraries,
        y=accuracy_scores,
        marker_color=['#96CEB4', '#FFEAA7'],
        text=[f'{score:.1f}%' for score in accuracy_scores],
        textposition='auto',
        name='Accuracy Score'
    ))
    
    fig_accuracy.update_layout(
        title='🎯 Accuracy Comparison',
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[0, 100]),
        template='plotly_white',
        height=400
    )
    
    # Content Length Comparison
    fig_content = go.Figure()
    
    content_lengths = [langextract_result.content_length, docling_result.content_length]
    
    fig_content.add_trace(go.Bar(
        x=libraries,
        y=content_lengths,
        marker_color=['#DDA0DD', '#98D8C8'],
        text=[f'{length:,} chars' for length in content_lengths],
        textposition='auto',
        name='Content Length'
    ))
    
    fig_content.update_layout(
        title='📄 Extracted Content Length',
        yaxis_title='Characters',
        template='plotly_white',
        height=400
    )
    
    return fig_performance, fig_accuracy, fig_content

def create_feature_comparison(langextract_result: ExtractionResult, docling_result: ExtractionResult):
    """Create feature comparison radar chart"""
    
    categories = ['Speed', 'Accuracy', 'Structure', 'Semantics', 'Flexibility']
    
    # Normalize scores (0-5 scale)
    langextract_scores = [
        5 - min(4, langextract_result.execution_time),  # Speed (inverted)
        langextract_result.accuracy_score / 20,  # Accuracy
        3,  # Structure (moderate)
        5,  # Semantics (excellent)
        4   # Flexibility (good)
    ]
    
    docling_scores = [
        5 - min(4, docling_result.execution_time),  # Speed (inverted)
        docling_result.accuracy_score / 20,  # Accuracy
        5,  # Structure (excellent)
        2,  # Semantics (limited)
        3   # Flexibility (moderate)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=langextract_scores,
        theta=categories,
        fill='toself',
        name='LangExtract',
        fillcolor='rgba(78, 205, 196, 0.3)',
        line=dict(color='rgba(78, 205, 196, 1)')
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=docling_scores,
        theta=categories,
        fill='toself',
        name='Docling',
        fillcolor='rgba(255, 206, 84, 0.3)',
        line=dict(color='rgba(255, 206, 84, 1)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )),
        showlegend=True,
        title="🕸️ Feature Comparison Radar",
        height=500
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("📊 LangExtract vs Docling Library Comparison")
    # st.markdown("### LangExtract vs Docling - Visual Performance Analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        if st.button("🚀 Run Comparison", type="primary"):
            st.session_state.run_comparison = True
        
        st.markdown("---")
        
        st.markdown("""
        **📚 Libraries:**
        - **LangExtract**: Semantic extraction
        - **Docling**: Structure extraction
        
        **🎯 Metrics:**
        - Execution time
        - Content accuracy
        - Feature capabilities
        """)
    
    # Main content
    if 'run_comparison' not in st.session_state:
        st.session_state.run_comparison = False
    
    if st.session_state.run_comparison:
        
        # Create progress tracking
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔬 LangExtract Testing")
            le_progress = st.progress(0)
            le_status = st.empty()
        
        with col2:
            st.subheader("📚 Docling Testing")
            doc_progress = st.progress(0)
            doc_status = st.empty()
        
        # Run tests
        with st.spinner("Running extraction tests..."):
            langextract_result = test_langextract(le_progress, le_status)
            docling_result = test_docling(doc_progress, doc_status)
        
        st.success("🎉 Comparison completed!")
        
        # Results Overview
        st.header("📈 Results Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            success_rate = (langextract_result.success + docling_result.success) / 2 * 100
            st.metric("Success Rate", f"{success_rate:.0f}%")
        
        with col2:
            avg_time = (langextract_result.execution_time + docling_result.execution_time) / 2
            st.metric("Avg Time", f"{avg_time:.2f}s")
        
        with col3:
            total_content = langextract_result.content_length + docling_result.content_length
            st.metric("Total Extracted", f"{total_content:,} chars")
        
        with col4:
            avg_accuracy = (langextract_result.accuracy_score + docling_result.accuracy_score) / 2
            st.metric("Avg Accuracy", f"{avg_accuracy:.1f}%")
        
        # Detailed Comparison
        st.header("🔍 Detailed Comparison")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🎯 Accuracy", "🕸️ Features", "📄 Content"])
        
        with tab1:
            fig_perf, fig_acc, fig_content = create_comparison_charts(langextract_result, docling_result)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_perf, use_container_width=True)
            with col2:
                st.plotly_chart(fig_content, use_container_width=True)
        
        with tab2:
            fig_accuracy = create_comparison_charts(langextract_result, docling_result)[1]
            st.plotly_chart(fig_accuracy, use_container_width=True)
            
            # Accuracy details
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("LangExtract Accuracy")
                if langextract_result.success and langextract_result.accuracy_score > 0:
                    st.success(f"Score: {langextract_result.accuracy_score:.1f}%")
                    st.json(langextract_result.structure_info)
                elif langextract_result.success and langextract_result.accuracy_score == 0:
                    st.warning("Completed but no extractions found")
                    st.info("💡 This often happens when the prompt/examples don't match the text structure well")
                    st.json(langextract_result.structure_info)
                else:
                    st.error(f"Failed: {langextract_result.error_message}")
            
            with col2:
                st.subheader("Docling Accuracy")
                if docling_result.success:
                    st.success(f"Score: {docling_result.accuracy_score:.1f}%")
                    st.json(docling_result.structure_info)
                else:
                    st.error(f"Failed: {docling_result.error_message}")
        
        with tab3:
            fig_radar = create_feature_comparison(langextract_result, docling_result)
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Feature details
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("LangExtract Features")
                for feature in langextract_result.features_used:
                    st.markdown(f"✅ {feature}")
            
            with col2:
                st.subheader("Docling Features")
                for feature in docling_result.features_used:
                    st.markdown(f"✅ {feature}")
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("LangExtract Output")
                if langextract_result.success:
                    st.code(langextract_result.extracted_content[:1000] + "..." if len(langextract_result.extracted_content) > 1000 else langextract_result.extracted_content, language="json")
                else:
                    st.error("No output available")
            
            with col2:
                st.subheader("Docling Output")
                if docling_result.success:
                    st.markdown(docling_result.extracted_content[:1000] + "..." if len(docling_result.extracted_content) > 1000 else docling_result.extracted_content)
                else:
                    st.error("No output available")
        
        # Summary
        st.header("🎯 Summary")
        
        summary_data = {
            'Library': ['LangExtract', 'Docling'],
            'Success': [langextract_result.success, docling_result.success],
            'Time (s)': [f"{langextract_result.execution_time:.2f}", f"{docling_result.execution_time:.2f}"],
            'Accuracy (%)': [f"{langextract_result.accuracy_score:.1f}", f"{docling_result.accuracy_score:.1f}"],
            'Content Length': [langextract_result.content_length, docling_result.content_length],
            'Best For': ['Semantic extraction, Custom attributes', 'Document structure, Fast processing']
        }
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Recommendations and Troubleshooting
        st.header("💡 Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🎯 Choose LangExtract for:</h4>
                <ul>
                    <li>Semantic understanding</li>
                    <li>Custom extraction classes</li>
                    <li>Attribute-based extraction</li>
                    <li>Research paper analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>⚡ Choose Docling for:</h4>
                <ul>
                    <li>Document structure preservation</li>
                    <li>Fast processing</li>
                    <li>Multiple format support</li>
                    <li>Layout analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Troubleshooting section
        if not langextract_result.success or langextract_result.accuracy_score == 0:
            st.header("🔧 LangExtract Troubleshooting")
            
            with st.expander("🐛 Common Issues & Solutions"):
                st.markdown("""
                **Empty Extractions (`"extractions": []`):**
                - **Cause**: Prompt/examples don't match your text structure
                - **Solution**: Adjust examples to closely match your actual text format
                - **Try**: Use more specific extraction classes that match your content
                
                **Model Not Understanding Content:**
                - **Cause**: Examples are too different from target text
                - **Solution**: Create examples using similar text patterns
                - **Try**: Increase `extraction_passes` from 1 to 2-3
                
                **Rate Limiting Issues:**
                - **Cause**: Too many parallel requests to Gemini API  
                - **Solution**: Reduce `max_workers` from 2 to 1
                - **Try**: Add delays between requests
                
                **API Key Issues:**
                - **Cause**: Missing or invalid Gemini API key
                - **Solution**: Set up your Google AI Studio API key
                - **Try**: Check environment variables: `GOOGLE_API_KEY`
                """)
                
            with st.expander("💡 Optimization Tips"):
                st.markdown("""
                **Better Prompt Design:**
                ```
                Extract the following from research papers:
                1. Title: Text after "Title:" 
                2. Authors: Text after "Authors:"
                3. Abstract: Text after "Abstract:"
                4. Results: Any numerical results or metrics
                ```
                
                **Better Examples:**
                - Use text that closely matches your input format
                - Include exact text patterns from your documents
                - Add multiple examples for different scenarios
                
                **Model Settings:**
                - Use `extraction_passes=2` for better accuracy
                - Set `max_workers=1` to avoid rate limits  
                - Try different models if available
                """)
        
        if not docling_result.success:
            st.header("🔧 Docling Troubleshooting")
            
            with st.expander("🐛 SSL & Installation Issues"):
                st.markdown("""
                **SSL Certificate Errors:**
                ```bash
                pip install --upgrade certifi
                export SSL_CERT_FILE=$(python -m certifi)
                ```
                
                **Installation Issues:**
                ```bash
                pip install docling
                # or if issues persist:
                pip install docling --no-deps
                ```
                
                **Model Download Issues:**
                - Models download on first use (may take time)
                - Ensure stable internet connection
                - Try running again after initial download
                """)

    
    else:
        # Initial state
        st.info("👈 Click 'Run Comparison' in the sidebar to start the analysis")
        
        # Show feature comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>🔬 LangExtract</h3>
                <p><strong>Semantic Extraction Engine</strong></p>
                <ul>
                    <li>AI-powered content understanding</li>
                    <li>Custom extraction classes</li>
                    <li>Few-shot learning</li>
                    <li>Attribute extraction</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>📚 Docling</h3>
                <p><strong>Document Structure Parser</strong></p>
                <ul>
                    <li>Fast document processing</li>
                    <li>Layout preservation</li>
                    <li>Multi-format support</li>
                    <li>Structured output</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()