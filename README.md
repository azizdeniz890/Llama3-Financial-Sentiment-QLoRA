# Llama-3 8B ile Finansal Metinler İçin Duygu Analizi (QLoRA ile İnce Ayar)

Bu proje, Üretken Yapay Zeka dersi kapsamında geliştirilmiş olup, Meta'nın Llama-3-8B-Instruct modelini kullanarak finansal haber başlıkları ve metinleri üzerinde duygu analizi yapmayı amaçlamaktadır. Model, QLoRA (Quantized Low-Rank Adaptation) tekniği ile "pozitif", "negatif" veya "nötr" duygu sınıflarını tanımak üzere ince ayarlanmıştır.

## Projeye Genel Bakış

Bu çalışmada, büyük bir dil modelinin (LLM) belirli bir görev için (finansal duygu analizi) nasıl daha verimli ve etkili bir şekilde eğitilebileceği gösterilmektedir. QLoRA yöntemi, modelin büyük bir kısmını 4-bit kuantizasyon ile dondurarak ve sadece küçük sayıda eğitilebilir LoRA adaptörlerini güncelleyerek bellek ve hesaplama kaynağı tasarrufu sağlar.

Proje, aşağıdaki ana bileşenleri içeren notebook'lar aracılığıyla sunulmaktadır:

1.  **`notebooks/ft1.ipynb`**:
    *   **Ana Eğitim Notebook'u:** Bu notebook, projenin temel adımlarını içerir:
        *   Gerekli Python kütüphanelerinin (PyTorch, Transformers, PEFT, TRL, bitsandbytes vb.) kurulumu.
        *   `meta-llama/Meta-Llama-3-8B-Instruct` temel modelinin Hugging Face Hub'dan indirilmesi/yüklenmesi ve 4-bit kuantizasyonu.
        *   Finansal duygu analizi için etiketlenmiş veri setinin (`data/data.csv`) yüklenmesi ve eğitim/değerlendirme/test setlerine ayrılması.
        *   **İnce ayar öncesi** temel modelin duygu analizi performansının değerlendirilmesi.
        *   QLoRA konfigürasyonlarının (LoRA katmanları, rank, alpha vb.) belirlenmesi.
        *   TRL kütüphanesinden `SFTTrainer` kullanılarak modelin ince ayarının yapılması.
        *   **İnce ayar sonrası** eğitilmiş modelin (adaptörlerle birlikte) performansının test seti üzerinde değerlendirilmesi.
        *   Eğitilmiş LoRA adaptörlerinin ve tokenizer'ın kaydedilmesi.

2.  **`notebooks/testft1.ipynb`**:
    *   **Detaylı Test ve Görselleştirme Notebook'u:** Bu notebook, `ft1.ipynb`'de eğitilen modelin performansını daha detaylı analiz eder ve görselleştirir:
        *   İnce ayar yapılmış modelin yüklenmesi.
        *   Test veri seti üzerinde tahminler yaparak doğruluk, kesinlik (precision), duyarlılık (recall) ve F1-skoru gibi metriklerin hesaplanması.
        *   Karmaşıklık matrislerinin (ince ayar öncesi ve sonrası) oluşturulması ve görselleştirilmesi.
        *   Eğitim sürecindeki kayıp (loss), öğrenme oranı (learning rate) ve gradyan normu gibi metriklerin grafiklerinin çizdirilmesi.
        *   Bu testler, modelin öğrenme sürecini ve farklı duygu sınıflarındaki performansını anlamamıza yardımcı olur.

3.  **`notebooks/easytest.ipynb` (veya `easytest_financial_sentiment.ipynb`)**:
    *   **Hızlı Test Notebook'u:** Bu notebook, eğitilmiş modelin ve adaptörlerin Hugging Face Hub'dan çekilerek kolayca test edilebilmesi için tasarlanmıştır.
        *   Kullanıcının sadece birkaç örnek finansal cümle girmesiyle modelin duygu tahminlerini hızlıca görmesini sağlar.
        *   Modelin pratik kullanımını ve erişilebilirliğini gösterir.

## Kullanılan Teknolojiler ve Kütüphaneler

*   **Model:** `meta-llama/Meta-Llama-3-8B-Instruct`
*   **İnce Ayar Tekniği:** QLoRA (Quantized Low-Rank Adaptation)
*   **Temel Kütüphaneler:**
    *   PyTorch
    *   Transformers (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline)
    *   PEFT (LoraConfig, PeftModel)
    *   TRL (SFTTrainer, SFTConfig)
    *   bitsandbytes (4-bit kuantizasyon için)
    *   Datasets
    *   scikit-learn (performans metrikleri)
    *   Pandas, NumPy (veri işleme)
    *   Matplotlib, Seaborn (görselleştirme)
    *   Hugging Face Hub (model ve adaptör paylaşımı)

## Kurulum ve Çalıştırma

1.  **Repository'yi Klonlayın:**
    ```bash
    git clone https://github.com/azizdeniz890/Llama3-Financial-Sentiment-QLoRA.git
    cd Llama3-Financial-Sentiment-QLoRA
    ```
2.  **Gerekli Kütüphaneleri Kurun:**
    Proje için gerekli Python kütüphanelerini `requirements.txt` dosyasını kullanarak kurun:
    ```bash
    pip install -r requirements.txt
    ```
    (Colab ortamında çalıştırıyorsanız, notebook'ların içindeki `!pip install` komutları bu işlemi yapacaktır.)
3.  **Notebook'ları Çalıştırın:**
    *   Eğitim sürecini ve detaylı analizleri görmek için `notebooks/ft1.ipynb` ve ardından `notebooks/testft1.ipynb` dosyalarını çalıştırın.
    *   Modeli hızlıca test etmek için `notebooks/easytest_financial_sentiment.ipynb` dosyasını çalıştırın.
    *   **Not:** Notebook'ları çalıştırırken (özellikle Colab'da) GPU (tercihen A100 veya benzeri) kullanmanız önerilir. Hugging Face Hub'dan model indirme/yükleme işlemleri için token gerekebilir.

## Veri Seti

Bu projede kullanılan veri seti `data/data.csv` dosyasında bulunmaktadır. Finansal metinler ve bunlara karşılık gelen "pozitif", "negatif" veya "nötr" duygu etiketlerini içerir.

## Sonuçlar ve Değerlendirme

Modelin performansı, ince ayar öncesi ve sonrası olmak üzere test veri seti üzerinde değerlendirilmiştir. Elde edilen metrikler ve görselleştirmeler aşağıdadır:

### İnce Ayar Öncesi Model Performansı

(Karmaşıklık Matrisi - İnce Ayar Öncesi)
![İnce Ayar Öncesi Karmaşıklık Matrisi](results/confusion_matrix_before.png)
*İnce ayar öncesi modelin test doğruluğu: ~%XX (Bu değeri notebook'unuzdan ekleyin)*

### İnce Ayar Sonrası Model Performansı

(Karmaşıklık Matrisi - İnce Ayar Sonrası)
![İnce Ayar Sonrası Karmaşıklık Matrisi](results/confusion_matrix_after.png)

(Sınıf Bazlı Precision, Recall, F1-Skoru - İnce Ayar Sonrası)
![İnce Ayar Sonrası Sınıf Bazlı Metrikler](results/fine_tuned_metrics_per_class.png)
*İnce ayar sonrası modelin test doğruluğu: ~%YY (Bu değeri notebook'unuzdan ekleyin)*

### Eğitim Süreci Grafikleri

Eğitim sırasında modelin öğrenme davranışı aşağıdaki grafiklerle izlenmiştir:

(Eğitim Kaybı ve Öğrenme Oranı)
![Eğitim Kaybı ve Öğrenme Oranı](results/training_loss_lr_epochs.png)

(Gradyan Normu)
![Gradyan Normu](results/gradient_norm_epochs.png)

İnce ayar sonrasında modelin finansal duygu analizi görevinde belirgin bir iyileşme gösterdiği gözlemlenmiştir.

## Adaptörlerin Kullanımı (Hugging Face Hub)

Eğitilmiş QLoRA adaptörleri, `azizdeniz890/Llama3-8B-Financial-Sentiment-LoRA` Hugging Face Hub reposundan çekilerek kullanılabilir. Detaylı kullanım örneği için `notebooks/easytest_financial_sentiment.ipynb` dosyasına bakınız.

## Katkıda Bulunan

*   Aziz Deniz ([azizdeniz890](https://github.com/azizdeniz890))
