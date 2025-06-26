# Projeyi localde çalıştırma adımları.

1.Sanal Ortam Oluşturma ve Gerekli Kütüphaneleri Kurma
Bu adım, projemizin bağımlılıklarını izole etmek ve doğru Python ortamında çalışmasını sağlamak için kritik öneme sahiptir. Sanal ortam, projemizin global Python kurulumuyla çakışmasını engeller.

# Yapılacaklar:

VS Code Terminalini Açın

VS Code'da üst menüden Terminal > New Terminal seçeneğine tıklayın veya Ctrl + Shift + ~ (tilde) kısayolunu kullanın. Terminal, VS Code'un alt panelinde açılacaktır. Proje klasörünüzde (FeatSel_Hyperparam_HIGGS) olduğundan emin olun.
Sanal Ortam Oluşturun:

Terminale aşağıdaki komutu yazın ve Enter'a basın
python -m venv .venv
Bu komut, proje dizininizin içinde .venv adında yeni bir sanal ortam oluşturacaktır. Bu işlem birkaç saniye sürebilir.

2.Sanal Ortamı Aktifleştirin:
İşletim sisteminize göre uygun komutu kullanın:
Windows (PowerShell): .venv\Scripts\Activate.ps1
Windows (Command Prompt / CMD): .venv\Scripts\activate.bat
Linux / macOS (Bash / Zsh): source .venv/bin/activate
