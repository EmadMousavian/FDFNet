# Getting Started

لطفا تصاویر داده های این پروژه را در پوشه به صورت زیر قرار دهید

از آنجایی که ما برای آموزش مدل تصاویر پنل های خورشیدی را به سلول های آن تقسیم کرده ایم، برای دانلود داده های آموزشی لطفا به لینک زیر مراجعه کنید

all-cell : [link](https://drive.google.com/file/d/1-tr2C-_XtE9Yxes-KDhLoPqkX3tEaY8n/view?usp=drive_link)

همچنین برای دانلود پوشه ترک های آموزشی برای آموزش و ارزیابی مدل نیز می توانید به لینک زیر مراجعه کنید

crack-img : [link](https://drive.google.com/file/d/1iCJRsHrgBflVO7QEIpq7gHY4CCzch5xp/view?usp=drive_link)

در زمان تست نیز تصاویر مورد نظر مسابقه به سلول های پنل تقسیم شده است، که برای سادگی ارزیابی و تست این بخش تصاویر سلول به سلول شده آن در پوشه های بخش تست قرار گرفته شده است


```
MicroCrack
├── data
│   ├── train
│   │   │── all-cell
│   │   │── crack-img
│   │   │   ├──clear_t & crack_t & crack_v
│   ├── test
│   │   │   ├──all_cell & clear & crack & pv_test_images
├── demo_data
├── model
```

* شما می توانید برای تست تصاویر خام پنل خورشیدی (یعنی سلول به سلول نشده) را در آدرس زیر قرار دهید

```commandline
data/test/pv_test_images
```

* لطفا با دستور زیر کتابخانه های مورد نظر و استفاده شده در این پروژه را نصب کنید
```python 
pip install -r requirements.txt
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

## TRAIN


* برای آموزش مدل کد زیر را اجرا کنید
```python 
python train.py --cfg_file config/DFT_DCMAF.yaml
```
نتایج این بخش در آدرس زیر ذخیره می شود

```commandline
output/DFT_DCMAF/default
```
 ``` config/DFT_DCMAF.yaml ``` در فایل مذکور اطلاعات آموزشی و معماری مدل و آدرس های داده ها قرار گرفته شده است


* در آدرس زیر نیز وزن های مدل آموزش دیده شده قرار گرفته شده است

```commandline
output/DFT_DCMAF/main/ckpt/best_f1_weighted.pth
```

* همچنین برای ادامه آموزش از یک وزن خاص با از یک دور آموزشی خاص می توانید آدرس وزن مربوطه و آخرین دور آموزشی که آن وزن بدست آمده را مطابق نمونه زیر قرار دهید 
```python 
python train.py --cfg_file config/DFT_DCMAF.yaml --batch_size 32 --ckpt output/DFT_DCMAF/default/ckpt/best_f1_weighted.pth --last_epoch 44
```


## TEST
برای تست مدل ابتدا باید تصاویر کامل پنل خورشیدی به سلول های آن تقسیم شود که برای این کار می توان از کد زیر استفاده کرد

```
python tools/data_prepare.py --data_path data/test/pv_test_images --output_path data/test/all_cell
```

* البته ما تصاویر سلول به سلول شده را در آدرس زیر قرار داده ایم

```commandline
data/test/all_cell
```

همچنین به صورت دستی بخشی از تصاویر سلول های تقسیم شده را به منظور ارزیابی مدل، بررسی و تصاویر تمیز و دارای کرک آن را به ترتیب در آدرس های زیر قرار دادیم
```commandline
data/test/clear
data/test/crack
```

پس از آماده سازی داده های تست می توان کد زیر را اجرا کرد تا نتایج مربوطه و ارسالی به مسابقه بدست آید

```python
python test.py --cfg_file config/DFT_DCMAF.yaml --ckpt output/DFT_DCMAF/main/ckpt/best_f1_weighted.pth --batch_size 1
```
نتایج این بخش در آدرس زیر قرار خواهد گرفت
```commandline
output/DFT_DCMAF/default/test_output
```

## DEMO

به منظور اجرا کد به صورت استنتاجی، کد زیر را قرار دادیم
این بخش تصاویر را به صورت کامل و خام (سلول به سلول نشده) می گیرد و در خروجی آن تصاویر سلول به سلول شده آن، فایل اکسل آدرس ردیف و ستون ترک های موجود و تصویر خروجی از سلول های دارای ترک در تصویر اصلی که به دور آن کادر قرمزی کشیده شده است را می دهد.

```python
python demo.py --data_path demo_data/PV_IMG_Sample
```

 ``` --data_path demo_data/PV_IMG_Sample ```

ورودی مذکور نشان دهنده آدرس پوشه دارای تصاویر خام پنل خورشیدی است


* در ورودی هم می توان آدرس پوشه که دارای تصاویر متعدد هست را داد و هم می توان آدرس تنها یک تصویر را داد
برای مثال

```python
python demo.py --data_path demo_data/PV_IMG_Sample    #  a directory of a folder
python demo.py --data_path "C:\Users\Mousavian\Desktop\Test data_microcrack\89.jpg"   #  a directory of just an image
```

* نمونه تصویر خروجی این بخش


![result_89.jpg](docs%2Fresult_89.jpg)