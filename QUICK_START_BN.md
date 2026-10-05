# NutriSight 1.1 — দ্রুত চালু করার নিয়ম

1. পুরোনো project-এর উপর file mix না করে এই folder-টা আলাদা করে extract করুন।
2. `backend\start.bat` চালান এবং terminal খোলা রাখুন।
3. backend terminal-এ `Your phone: http://...:8000` দেখাবে।
4. নতুন terminal-এ `frontend\start.bat` চালান।
5. Expo Go দিয়ে QR scan করুন।
6. app নিজে থেকেই Expo চালানো PC-এর IP ধরে backend connect করার চেষ্টা করবে।
7. connection না হলে NutriSight -> Settings -> Backend connection-এ backend terminal-এর `http://...:8000` address দিন।

## আগের error কেন হয়েছিল

Physical phone-এ `localhost` / `127.0.0.1` মানে PC নয়, phone নিজেই। তাই `Failed to connect to localhost/127.0.0.1:8000` হচ্ছিল। Version 1.1 native device-এ পুরোনো localhost setting automatically বদলে Expo development PC-এর address ব্যবহার করে।

## Backend browser check

Phone বা PC browser থেকে:

`http://YOUR-PC-IP:8000/`

এখন `detail: Not Found` না এসে NutriSight API status JSON দেখাবে।
