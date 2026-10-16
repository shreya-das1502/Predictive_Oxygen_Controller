import subprocess,time,urllib.request
p=subprocess.Popen(['python','-m','streamlit','run','dashboard/app.py','--global.developmentMode=false','--server.headless=true','--server.address=127.0.0.1','--server.port=8524','--browser.gatherUsageStats=false'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
try:
 for _ in range(40):
  if p.poll() is not None: raise RuntimeError('server exited')
  try:
   with urllib.request.urlopen('http://127.0.0.1:8524/_stcore/health',timeout=1) as r:
    assert r.status==200
    print('HEALTH',r.status,r.read().decode());break
  except OSError: time.sleep(.5)
 else: raise RuntimeError('health timeout')
finally:
 p.terminate()
 try: out=p.communicate(timeout=10)[0]
 except subprocess.TimeoutExpired: p.kill();out=p.communicate()[0]
 print(out)
