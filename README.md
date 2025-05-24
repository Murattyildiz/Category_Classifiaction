# Türkçe Metin Kategori Sınıflandırıcı

Bu uygulama, Türkçe metinleri 7 farklı kategoriye sınıflandıran bir yapay zeka modeli için web arayüzüdür.

## Kategoriler
- **Dünya**: Uluslararası haberler ve gelişmeler
- **Ekonomi**: Ekonomik haberler, borsa, finans
- **Kültür**: Sanat, kültür, festival haberleri
- **Sağlık**: Sağlık, tıp, pandemi haberleri
- **Siyaset**: Siyasi gelişmeler ve haberler
- **Spor**: Spor haberleri ve sonuçları
- **Teknoloji**: Teknoloji, yapay zeka, inovasyon

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Uygulamayı çalıştırın:
```bash
streamlit run app.py
```

## Kullanım

1. Web tarayıcınızda otomatik olarak açılacak olan arayüze gidin
2. Sol tarafta bulunan metin giriş alanına analiz etmek istediğiniz Türkçe metni yazın
3. Hızlı test için örnek metinlere tıklayabilirsiniz
4. "Kategoriyi Tahmin Et" butonuna tıklayın
5. Sağ tarafta sonuçları göreceksiniz:
   - Tahmin edilen kategori
   - Güven skoru (%)
   - Tüm kategoriler için detaylı skorlar
   - Görsel grafik

## Özellikler

- **Modern Web Arayüzü**: Kullanıcı dostu, responsive tasarım
- **Gerçek Zamanlı Tahmin**: Hızlı ve doğru kategori tahmini
- **Görsel Analiz**: Plotly ile interaktif grafikler
- **Örnek Metinler**: Hızlı test için hazır örnekler
- **Detaylı Sonuçlar**: Tüm kategoriler için güven skorları

## Teknik Detaylar

- **Framework**: Streamlit
- **Model**: TensorFlow/Keras Transformer tabanlı model
- **Maksimum Metin Uzunluğu**: 375 token
- **Görselleştirme**: Plotly
- **Dil**: Türkçe

## Kullandığım Modelin linki:https://www.kaggle.com/code/muratyilldiz/nlp-text/
