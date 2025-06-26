# Projeyi localde çalıştırma adımları.

1.Sanal Ortam Oluşturma ve Gerekli Kütüphaneleri Kurma
Bu adım, projemizin bağımlılıklarını izole etmek ve doğru Python ortamında çalışmasını sağlamak için kritik öneme sahiptir. Sanal ortam, projemizin global Python kurulumuyla çakışmasını engeller.

# Yapılacaklar:

VS Code Terminalini Açın

VS Code'da üst menüden Terminal > New Terminal seçeneğine tıklayın veya Ctrl + Shift + ~ (tilde) kısayolunu kullanın. Terminal, VS Code'un alt panelinde açılacaktır. Proje klasörünüzde (FeatSel_Hyperparam_HIGGS) olduğundan emin olun.
Sanal Ortam Oluşturun:

Terminale aşağıdaki komutu yazın ve Enter'a basın

python -m venv .venv =>
Bu komut, proje dizininizin içinde .venv adında yeni bir sanal ortam oluşturacaktır. Bu işlem birkaç saniye sürebilir.

2.Sanal Ortamı Aktifleştirin:

İşletim sisteminize göre uygun komutu kullanın:
Windows (PowerShell): .venv\Scripts\Activate.ps1

Windows (Command Prompt / CMD): .venv\Scripts\activate.bat

Linux / macOS (Bash / Zsh): source .venv/bin/activate

Sanal ortam başarıyla aktifleştirildiğinde, terminal prompt'unuzun başında (.venv) ifadesini görmelisiniz. Bu, artık sanal ortamın içinde çalıştığınızı gösterir.

![image](https://github.com/user-attachments/assets/bb2201e1-05ce-43af-9150-108d01943a92)

Gerekli Kütüphaneleri Kurun:

Sanal ortam aktifleştirildikten sonra, requirements.txt dosyasında listelediğimiz tüm kütüphaneleri kurmak için aşağıdaki komutu çalıştırın:

pip install -r requirements.txt => Bu komut, numpy, pandas, scikit-learn, matplotlib, seaborn ve xgboost kütüphanelerini indirecek ve sanal ortamınıza kuracaktır. Bu işlem internet hızınıza ve kütüphane boyutlarına bağlı olarak biraz zaman alabilir.

VS Code'da Python Yorumlayıcısını Seçin:

Bu önemli bir adımdır! VS Code'un bu projeniz için yeni oluşturduğunuz sanal ortamı kullanmasını sağlamalısınız.
VS Code'un sağ alt köşesindeki durum çubuğunda (genellikle "Python x.x.x" yazar) tıklayın.

Açılan komut paletinde, Enter interpreter path... seçeneğini veya doğrudan .venv klasörünüzün içindeki Python yorumlayıcısını (.venv/bin/python veya .venv\Scripts\python.exe) seçin.

Eğer listede görünmüyorsa, "Enter interpreter path..." seçeneğini seçip yolunu elle girmeniz gerekebilir. VS Code genellikle yeni oluşturulan sanal ortamı otomatik olarak algılar.

Tamamlandığında:

Terminalinizde prompt başında (.venv) görmelisiniz.

pip install -r requirements.txt komutu başarıyla tamamlanmış olmalı.
VS Code'un yorumlayıcısı .venv sanal ortamına ayarlanmış olmalı.

# Veri Setini İndirme ve src/preprocess.py Dosyasını Oluşturma
Bu adımda, projemizin kullanacağı HIGGS veri setini indireceğiz ve veri ön işleme fonksiyonlarını içerecek src/preprocess.py dosyasını oluşturup ilk içeriğini ekleyeceğiz.

Yapılacaklar:

1- HIGGS Veri Setini İndirme:

HIGGS veri seti çok büyüktür (yaklaşık 11 milyon örnek). Doğrudan GitHub'a yüklenmesi önerilmez. Genellikle manuel olarak indirilir veya bir indirme betiği kullanılır.

Verilen linkten (https://archive.ics.uci.edu/ml/datasets/HIGGS) veri setini indirin. Sayfada "Data Folder" linkine tıklayıp HIGGS.csv.gz dosyasını indirin.

İndirdiğiniz HIGGS.csv.gz dosyasını projenizin FeatSel_Hyperparam_HIGGS/data/ klasörünün içine taşıyın. Aşağıdaki olması gerekir.

![image](https://github.com/user-attachments/assets/1ca9f1d1-8452-4172-89df-39d449535bb0)

2- src/preprocess.py Dosyasını Oluşturma ve İçeriğini Ekleme:

src klasörünün içine preprocess.py adında yeni bir Python dosyası oluşturun.

dosyayı oluşturduktan sonra sanal ortama geçtiğinizden emin olun. aşağıdaki resimde Python 3.11.9(.venv) yazan kısma tıklamanız gerekiyor. 

![image](https://github.com/user-attachments/assets/e6abd1a5-7fa8-4433-9dca-4df82788e7e2)


Bu dosyaya, veri ön işleme adımlarını (özellikle aykırı değer tespiti ve işleme) gerçekleştirecek fonksiyonları ekleyeceğiz. MinMaxScaler ise pipeline içinde kullanılacağı için burada ayrı bir fonksiyon olarak yazmayacağız.

NOT: preprocess.py kodları repoda mevcutur.

3- src/__init__.py Dosyasını Oluşturma:

src klasörünün içine boş bir dosya oluşturun ve adını __init__.py koyun.

Bu boş dosya, Python'a src klasörünün bir paket olduğunu bildirir ve main.py gibi diğer dosyalardan preprocess.py gibi modülleri kolayca içe aktarmanızı sağlar (örneğin from src.preprocess import preprocess_data).

# Taslağı Ekleme
src/main.py Dosyasını Oluşturma ve İlk Taslağı Ekleme

Bu adımda, projemizin ana akışını yönetecek olan main.py dosyasını oluşturacağız. Bu dosya, veri yüklemeden başlayarak ön işleme, modelleme ve değerlendirme adımlarını orkestra edecek.

Yapılacaklar:

src/main.py Dosyasını Oluşturma:

src klasörünün içine main.py adında yeni bir Python dosyası oluşturun.
main.py Dosyasının İçeriğini Ekleme: repodaki main.py içindeki kodları alıp oluşturduğunuz main.py dosyasına ekleyin ve kaydein.

Bu kod, veri yükleme, preprocess.py'den ön işleme fonksiyonunu çağırma ve genel pipeline yapısının taslağını içerir.

Kodu Çalıştırma Testi (İsteğe Bağlı):

Bu aşamada, kodunuzun veri setini başarıyla yükleyip ön işleme fonksiyonunu çağırdığından emin olmak için bir test çalıştırabilirsiniz.

VS Code terminalinde (sanal ortamınızın aktif olduğundan emin olun): (python src/main.py) bu komutu girerek (powershell) aşağıdaki gibi bir görüntü oluşması gerekir.

![image](https://github.com/user-attachments/assets/cae44edd-587f-4e60-9e71-6e6e663bde48)

# Sonuç : ROC Eğrileri Karşılaştırması
![image](https://github.com/user-attachments/assets/02849b92-ea86-456b-b115-0781253d8c4b)




