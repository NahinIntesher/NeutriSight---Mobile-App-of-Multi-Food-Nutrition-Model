# NutriSight Frontend চালানোর নিয়ম

এই version-এ আপনার দেওয়া reference image-এর UI flow frontend-এ rebuild করা হয়েছে।

1. পুরোনো frontend folder backup রাখুন।
2. এই ZIP unzip করে পাওয়া `frontend` folder দিয়ে পুরোনো frontend replace করুন। Merge না করাই ভালো।
3. প্রথমবার `setup.bat` চালান।
4. তারপর `start.bat` চালান।
5. Physical phone ব্যবহার করলে Settings → Backend Connection থেকে PC-এর LAN IP দিন, যেমন `http://192.168.0.25:8000`।

মূল flow:
Splash → Smart Nutrition onboarding → Login/Signup → Profile setup → Home → Scan → Analyzing → Detected Items → Nutrition Results → Nutrition Insights

History, Profile, Settings এবং backend status-ও connected আছে।
