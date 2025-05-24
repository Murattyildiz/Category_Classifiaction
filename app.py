import streamlit as st
import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Türkçe Metin Kategori Sınıflandırıcı",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stil
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-score {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
    }
    .category-label {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_components():
    """Model ve yardımcı bileşenleri yükle"""
    try:
        # Keras 3 için TFSMLayer kullanarak modeli yükle
        try:
            # Önce normal load_model ile deneyelim
            model = tf.keras.models.load_model('turkish_model_saved/')
        except Exception as e:
            # Eğer başarısız olursa TFSMLayer kullan
            print(f"Normal yükleme başarısız, TFSMLayer kullanılıyor: {e}")
            model = tf.keras.layers.TFSMLayer('turkish_model_saved/', call_endpoint='serving_default')
        
        # Tokenizer'ı yükle
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        # Label encoder'ı yükle
        with open('label_encoder.pickle', 'rb') as handle:
            le = pickle.load(handle)
        
        # Model konfigürasyonunu yükle
        with open('model_config.json', 'r', encoding='utf-8') as f:
            model_config = json.load(f)
        
        return model, tokenizer, le, model_config
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None, None, None, None

def preprocess_text(text):
    """Metni ön işlemden geçir"""
    # Basit temizleme
    text = text.lower().strip()
    # Gereksiz boşlukları temizle
    text = ' '.join(text.split())
    return text

def predict_text_category(text, model, tokenizer, model_config):
    """Metnin kategorisini tahmin et"""
    try:
        # Metin ön işleme
        processed_text = preprocess_text(text)
        
        # Tokenize ve pad işlemleri
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=model_config['maxlen'])
        
        # Model tipine göre tahmin yap
        if hasattr(model, 'predict'):
            # Normal Keras model
            prediction = model.predict(padded_sequence, verbose=0)[0]
        else:
            # TFSMLayer kullanılıyorsa - int64 olarak dönüştür
            input_tensor = tf.convert_to_tensor(padded_sequence, dtype=tf.int64)
            prediction_dict = model(input_tensor)
            
            # TFSMLayer çıktısı dict olabilir, doğru anahtarı bulalım
            if isinstance(prediction_dict, dict):
                # En muhtemel anahtar isimlerini deneyelim
                possible_keys = ['output_0', 'dense', 'predictions', 'logits', 'output']
                prediction = None
                for key in possible_keys:
                    if key in prediction_dict:
                        prediction = prediction_dict[key].numpy()[0]
                        break
                if prediction is None:
                    # Eğer hiçbir anahtar bulunamazsa, ilk değeri al
                    first_key = list(prediction_dict.keys())[0]
                    prediction = prediction_dict[first_key].numpy()[0]
            else:
                prediction = prediction_dict.numpy()[0]
        
        category_index = np.argmax(prediction)
        
        # Kategori isimlerini temizle (boşlukları kaldır)
        clean_classes = [cls.strip() for cls in model_config['classes']]
        category = clean_classes[category_index]
        confidence = float(prediction[category_index])
        
        # Tüm skorları hazırla
        all_scores = dict(zip(clean_classes, prediction.tolist()))
        
        return {
            "category": category,
            "confidence": confidence,
            "all_scores": all_scores
        }
    except Exception as e:
        st.error(f"Tahmin sırasında hata oluştu: {e}")
        return None

def create_confidence_chart(scores):
    """Güven skorları için grafik oluştur"""
    df = pd.DataFrame(list(scores.items()), columns=['Kategori', 'Skor'])
    df = df.sort_values('Skor', ascending=True)
    
    fig = px.bar(
        df, 
        x='Skor', 
        y='Kategori',
        orientation='h',
        title='Tüm Kategoriler için Güven Skorları',
        color='Skor',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    # Ana başlık
    st.markdown('<h1 class="main-header">📝 Türkçe Metin Kategori Sınıflandırıcı</h1>', unsafe_allow_html=True)
    
    # Model ve bileşenleri yükle
    model, tokenizer, le, model_config = load_model_and_components()
    
    if model is None:
        st.error("Model yüklenemedi. Lütfen dosyaların doğru konumda olduğundan emin olun.")
        return
    
    # Sidebar bilgileri
    with st.sidebar:
        st.header("ℹ️ Model Bilgileri")
        st.write(f"**Maksimum Metin Uzunluğu:** {model_config['maxlen']}")
        st.write(f"**Kategori Sayısı:** {len(model_config['classes'])}")
        st.write("**Kategoriler:**")
        for i, cat in enumerate([cls.strip() for cls in model_config['classes']], 1):
            st.write(f"{i}. {cat.title()}")
    
    # Ana içerik alanı
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 Metin Girişi")
        
        # Örnek metinler
        example_texts = {
            "Ekonomi": "Borsa İstanbul'da hisseler yükselişe geçerken, dolar kuru da düşüş gösterdi.",
            "Spor": "Galatasaray, Fenerbahçe derbisinde 2-1 galip gelerek liderliğini sürdürdü.",
            "Teknoloji": "Yapay zeka teknolojileri ile geliştirilen yeni uygulama büyük ilgi gördü.",
            "Sağlık": "Koronavirüs aşısının üçüncü dozu için randevu sistemi başlatıldı.",
            "Siyaset": "Cumhurbaşkanı, yeni ekonomi politikalarını açıklayacağını duyurdu.",
            "Kültür": "İstanbul Film Festivali'nde en iyi film ödülü Türk sinemasına gitti.",
            "Dünya": "Avrupa Birliği ülkeleri iklim değişikliği konusunda yeni anlaşma imzaladı."
        }
        
        st.write("**Örnek metinler:** (Tıklayarak kullanabilirsiniz)")
        example_cols = st.columns(4)
        
        selected_example = None
        for i, (category, text) in enumerate(example_texts.items()):
            col_idx = i % 4
            with example_cols[col_idx]:
                if st.button(f"{category}", key=f"example_{i}"):
                    selected_example = text
        
        # Metin giriş alanı
        default_text = selected_example if selected_example else ""
        user_text = st.text_area(
            "Kategorisini öğrenmek istediğiniz metni girin:",
            value=default_text,
            height=150,
            placeholder="Buraya Türkçe metin yazın..."
        )
        
        # Tahmin butonu
        predict_button = st.button("🔍 Kategoriyi Tahmin Et", type="primary")
    
    with col2:
        st.header("📊 Sonuçlar")
        
        if predict_button and user_text.strip():
            with st.spinner("Tahmin yapılıyor..."):
                result = predict_text_category(user_text, model, tokenizer, model_config)
            
            if result:
                # Ana tahmin sonucu
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="category-label">Kategori: {result['category'].title()}</div>
                    <div class="confidence-score">Güven: %{result['confidence']*100:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Güven seviyesi göstergesi
                confidence_color = "green" if result['confidence'] > 0.7 else "orange" if result['confidence'] > 0.5 else "red"
                st.progress(result['confidence'])
                
                # Detaylı skorlar
                st.subheader("📈 Detaylı Skorlar")
                
                # Grafik
                fig = create_confidence_chart(result['all_scores'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Tablo
                scores_df = pd.DataFrame(
                    [(k.title(), f"%{v*100:.2f}") for k, v in sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True)],
                    columns=['Kategori', 'Skor']
                )
                st.dataframe(scores_df, use_container_width=True)
        
        elif predict_button and not user_text.strip():
            st.warning("⚠️ Lütfen tahmin için bir metin girin.")
    
    # Alt bilgi
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🤖 Bu uygulama, Türkçe metinleri 7 farklı kategoriye sınıflandıran bir yapay zeka modeli kullanmaktadır.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()