import os
import io
import base64

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']  = '-1'

import warnings
warnings.filterwarnings('ignore')

import numpy as np # type: ignore
from flask import Flask, request, jsonify, render_template_string # type: ignore
from PIL import Image, ImageDraw, ImageFont # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore

MODEL_PATH = '../models/currency_model.h5'
IMG_SIZE   = (224, 224)

app = Flask(__name__)
print("[INFO] Loading model...")
model = load_model(MODEL_PATH)
print("[SUCCESS] Model ready.")

HTML = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>نظام كشف العملات المزيفة</title>
<link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#09101f;--surface:#111827;--card:#141e30;--border:#1e3358;--gold:#c9a84c;--gold-l:#e8c97a;--green:#22c55e;--red:#ef4444;--text:#e2e8f0;--muted:#64748b}
body{font-family:'Tajawal',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;flex-direction:column;align-items:center}
body::before{content:'';position:fixed;inset:0;pointer-events:none;background-image:linear-gradient(rgba(201,168,76,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(201,168,76,.03) 1px,transparent 1px);background-size:44px 44px}
header{width:100%;padding:22px 40px;display:flex;align-items:center;gap:14px;border-bottom:1px solid var(--border);background:rgba(17,24,39,.9);backdrop-filter:blur(14px);position:sticky;top:0;z-index:10}
.logo{width:42px;height:42px;background:linear-gradient(135deg,var(--gold),var(--gold-l));border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px;box-shadow:0 0 20px rgba(201,168,76,.25)}
.logo-text h1{font-size:19px;font-weight:800;color:var(--gold-l)}.logo-text p{font-size:12px;color:var(--muted);margin-top:2px}
.badge{margin-right:auto;background:rgba(201,168,76,.1);border:1px solid rgba(201,168,76,.25);color:var(--gold-l);font-size:12px;padding:4px 14px;border-radius:20px}
main{width:100%;max-width:920px;padding:44px 20px;display:flex;flex-direction:column;gap:28px}
.upload-card{background:var(--card);border:2px dashed var(--border);border-radius:18px;padding:52px 32px;text-align:center;cursor:pointer;transition:border-color .3s,background .3s}
.upload-card:hover,.upload-card.drag{border-color:var(--gold);background:rgba(201,168,76,.04)}
.upload-icon{font-size:54px;margin-bottom:14px}
.upload-card h2{font-size:20px;font-weight:700;margin-bottom:8px}.upload-card p{font-size:14px;color:var(--muted)}
#fileInput{display:none}
.btn-upload{margin-top:20px;padding:12px 34px;background:linear-gradient(135deg,var(--gold),var(--gold-l));color:#09101f;font-family:'Tajawal',sans-serif;font-size:15px;font-weight:700;border:none;border-radius:10px;cursor:pointer;transition:opacity .2s,transform .2s}
.btn-upload:hover{opacity:.88;transform:translateY(-1px)}
#previewSection{display:none}
.top-bar{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.section-title{font-size:16px;font-weight:700}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:600px){.grid2{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--border);border-radius:14px;overflow:hidden}
.panel-hdr{padding:12px 18px;border-bottom:1px solid var(--border);font-size:12px;font-weight:600;color:var(--muted);letter-spacing:.4px}
.panel img{width:100%;max-height:260px;object-fit:contain;display:block;background:#0b1220}
.btn-analyse{width:100%;padding:15px;margin-top:18px;background:linear-gradient(135deg,#163566,#1e4d8c);border:1px solid var(--border);color:var(--text);font-family:'Tajawal',sans-serif;font-size:16px;font-weight:700;border-radius:12px;cursor:pointer;transition:all .3s;display:flex;align-items:center;justify-content:center;gap:10px}
.btn-analyse:hover:not(:disabled){background:linear-gradient(135deg,#1e4d8c,#2563ae);transform:translateY(-1px)}
.btn-analyse:disabled{opacity:.45;cursor:not-allowed}
.btn-reset{padding:9px 22px;background:transparent;border:1px solid var(--border);color:var(--muted);font-family:'Tajawal',sans-serif;font-size:13px;border-radius:8px;cursor:pointer;transition:all .2s}
.btn-reset:hover{border-color:var(--gold);color:var(--gold-l)}
.loader{display:none;align-items:center;justify-content:center;gap:12px;color:var(--muted);font-size:14px;margin-top:14px}
.spinner{width:20px;height:20px;border:2px solid var(--border);border-top-color:var(--gold);border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
#resultSection{display:none}
.result-card{border-radius:16px;border:1px solid var(--border);overflow:hidden}
.result-hdr{padding:22px 26px;display:flex;align-items:center;gap:16px}
.result-hdr.authentic{background:rgba(34,197,94,.1);border-bottom:1px solid rgba(34,197,94,.22)}
.result-hdr.counterfeit{background:rgba(239,68,68,.1);border-bottom:1px solid rgba(239,68,68,.22)}
.result-icon{font-size:42px}.result-label{font-size:22px;font-weight:800}
.result-label.green{color:var(--green)}.result-label.red{color:var(--red)}
.result-sub{font-size:13px;color:var(--muted);margin-top:4px}
.result-body{padding:24px;background:var(--card);display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:480px){.result-body{grid-template-columns:1fr}}
.stat-box{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;text-align:center}
.stat-lbl{font-size:11px;color:var(--muted);letter-spacing:.5px;margin-bottom:8px}
.stat-val{font-size:34px;font-weight:800}
.stat-val.green{color:var(--green)}.stat-val.red{color:var(--red)}.stat-val.gold{color:var(--gold-l)}
.bar-wrap{background:rgba(255,255,255,.06);border-radius:6px;height:7px;margin-top:12px;overflow:hidden}
.bar-fill{height:100%;border-radius:6px;transition:width 1.1s ease}
.bar-fill.green{background:linear-gradient(90deg,#16a34a,var(--green))}.bar-fill.red{background:linear-gradient(90deg,#b91c1c,var(--red))}
footer{margin-top:auto;padding:22px;text-align:center;font-size:12px;color:var(--muted);border-top:1px solid var(--border);width:100%}
</style>
</head>
<body>
<header>
  <div class="logo">💵</div>
  <div class="logo-text">
    <h1>نظام كشف العملات المزيفة</h1>
    <p>Currency Counterfeit Detection System</p>
  </div>
  <span class="badge">MobileNetV2 · Accuracy 95.21%</span>
</header>
<main>
  <div class="upload-card" id="dropZone">
    <div class="upload-icon">🏦</div>
    <h2>ارفع صورة العملة للفحص</h2>
    <p>Upload a banknote image to analyse</p>
    <p style="margin-top:8px;font-size:12px;color:var(--muted)">JPG · PNG · JPEG</p>
    <button class="btn-upload" onclick="document.getElementById('fileInput').click()">📂 &nbsp;اختر صورة · Choose Image</button>
    <input type="file" id="fileInput" accept="image/*"/>
  </div>
  <div id="previewSection">
    <div class="top-bar">
      <span class="section-title">معاينة الصورة · Image Preview</span>
      <button class="btn-reset" onclick="resetAll()">↩ إعادة تعيين</button>
    </div>
    <div class="grid2">
      <div class="panel">
        <div class="panel-hdr">📷 الصورة الأصلية · Original Image</div>
        <img id="previewImg" src="" alt="preview"/>
      </div>
      <div class="panel" id="resultImgPanel" style="display:none">
        <div class="panel-hdr">🔍 نتيجة التحليل · Analysis Result</div>
        <img id="resultImg" src="" alt="result"/>
      </div>
    </div>
    <button class="btn-analyse" id="analyseBtn" onclick="analyse()">🔬 &nbsp;تحليل العملة &nbsp;·&nbsp; Analyse Currency</button>
    <div class="loader" id="loader">
      <div class="spinner"></div>
      <span>جاري التحليل... · Analysing, please wait</span>
    </div>
  </div>
  <div id="resultSection">
    <div class="result-card">
      <div class="result-hdr" id="resultHdr">
        <div class="result-icon" id="resultIcon"></div>
        <div>
          <div class="result-label" id="resultLabel"></div>
          <div class="result-sub" id="resultSub"></div>
        </div>
      </div>
      <div class="result-body">
        <div class="stat-box">
          <div class="stat-lbl">CONFIDENCE · نسبة الثقة</div>
          <div class="stat-val" id="confVal"></div>
          <div class="bar-wrap"><div class="bar-fill" id="confBar" style="width:0%"></div></div>
        </div>
        <div class="stat-box">
          <div class="stat-lbl">STATUS · الحالة</div>
          <div class="stat-val gold" id="statusVal" style="font-size:20px;margin-top:10px;line-height:1.3"></div>
        </div>
      </div>
    </div>
  </div>
</main>
<footer>نظام كشف العملات المزيفة · Currency Detection System v4.0 &nbsp;|&nbsp; MobileNetV2 Transfer Learning</footer>
<script>
let selectedFile=null;
const dropZone=document.getElementById('dropZone');
dropZone.addEventListener('dragover',e=>{e.preventDefault();dropZone.classList.add('drag')});
dropZone.addEventListener('dragleave',()=>dropZone.classList.remove('drag'));
dropZone.addEventListener('drop',e=>{e.preventDefault();dropZone.classList.remove('drag');const f=e.dataTransfer.files[0];if(f&&f.type.startsWith('image/'))handleFile(f)});
document.getElementById('fileInput').addEventListener('change',e=>{if(e.target.files[0])handleFile(e.target.files[0])});
function handleFile(file){selectedFile=file;const r=new FileReader();r.onload=ev=>{document.getElementById('previewImg').src=ev.target.result;document.getElementById('previewSection').style.display='block';document.getElementById('resultSection').style.display='none';document.getElementById('resultImgPanel').style.display='none';document.getElementById('dropZone').style.display='none'};r.readAsDataURL(file)}
async function analyse(){if(!selectedFile)return;const btn=document.getElementById('analyseBtn');const loader=document.getElementById('loader');btn.disabled=true;loader.style.display='flex';const fd=new FormData();fd.append('image',selectedFile);try{const res=await fetch('/predict',{method:'POST',body:fd});const data=await res.json();if(data.error){alert('خطأ: '+data.error);return}showResult(data)}catch(e){alert('حدث خطأ في الاتصال بالخادم')}finally{btn.disabled=false;loader.style.display='none'}}
function showResult(data){const isReal=data.status==='AUTHENTIC';const cls=isReal?'authentic':'counterfeit';const color=isReal?'green':'red';document.getElementById('resultHdr').className='result-hdr '+cls;document.getElementById('resultIcon').textContent=isReal?'✅':'❌';document.getElementById('resultLabel').className='result-label '+color;document.getElementById('resultLabel').textContent=isReal?'عملة حقيقية · Authentic':'عملة مزيفة · Counterfeit';document.getElementById('resultSub').textContent=isReal?'العملة أصلية وصالحة للتداول · The currency appears genuine':'تحذير: العملة مشبوهة · Warning: Possible counterfeit detected';const pct=(data.confidence*100).toFixed(2)+'%';document.getElementById('confVal').textContent=pct;document.getElementById('confVal').className='stat-val '+color;document.getElementById('statusVal').textContent=data.status;const bar=document.getElementById('confBar');bar.className='bar-fill '+color;setTimeout(()=>bar.style.width=(data.confidence*100)+'%',80);if(data.result_image){document.getElementById('resultImg').src='data:image/png;base64,'+data.result_image;document.getElementById('resultImgPanel').style.display='block'}document.getElementById('resultSection').style.display='block';document.getElementById('resultSection').scrollIntoView({behavior:'smooth'})}
function resetAll(){selectedFile=null;document.getElementById('fileInput').value='';document.getElementById('previewSection').style.display='none';document.getElementById('resultSection').style.display='none';document.getElementById('dropZone').style.display='block'}
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'لم يتم إرسال صورة'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400
    try:
        img_bytes   = file.read()
        img         = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_display = img.copy().resize((600, 600))
        img_model   = img.resize(IMG_SIZE)
        x = np.array(img_model, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        prediction = float(model.predict(x, verbose=0).flatten()[0])
        if prediction > 0.5:
            status, label, confidence, border_rgb = 'AUTHENTIC', 'Real Currency ($)', prediction, (34,197,94)
        else:
            status, label, confidence, border_rgb = 'COUNTERFEIT', 'Fake Currency (!)', 1-prediction, (239,68,68)
        result_img = img_display.copy()
        draw = ImageDraw.Draw(result_img)
        draw.rectangle([0,0,result_img.width-1,result_img.height-1], outline=border_rgb, width=10)
        draw.rectangle([0,0,result_img.width,58], fill=(*border_rgb,230))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
        draw.text((12,16), f"{label}  |  Confidence: {confidence*100:.2f}%", fill=(255,255,255), font=font)
        buf = io.BytesIO()
        result_img.save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return jsonify({'status':status,'label':label,'confidence':confidence,'result_image':img_b64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*45)
    print("  Currency Detection Web App")
    print("  http://127.0.0.1:5000")
    print("="*45 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)