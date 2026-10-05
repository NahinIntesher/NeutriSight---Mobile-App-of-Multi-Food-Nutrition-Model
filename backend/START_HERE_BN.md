# NutriSight Backend চালু করার নিয়ম

1. `backend`, `frontend`, `models` তিনটা folder একই parent folder-এর ভিতরে রাখুন।
2. আপনার existing model folder-এ এই তিনটা file থাকতে হবে:
   - `models/yolo/best.pt`
   - `models/vit_lstm/best_model.pth`
   - `models/nutrition/best_model.pth`
3. প্রথমবার `backend/setup.bat` চালান।
4. এরপর প্রতিবার `backend/start.bat` চালালেই API `0.0.0.0:8000`-এ চালু হবে।
5. Browser-এ `http://127.0.0.1:8000/health` খুলে model status দেখতে পারবেন।

### গুরুত্বপূর্ণ পরিবর্তন

- ViT fallback এখন চালু: YOLO confidence 70%-এর নিচে হলে ViT check করবে, ViT 60% বা তার বেশি হলে label গ্রহণ করবে।
- Expo-এর `Unsupported FormDataPart implementation` এড়াতে mobile app raw image upload endpoint ব্যবহার করে।
- নতুন user প্রথম login/signup-এর পর Meal Insights profile form দেখবে। চাইলে **Skip for now** করতে পারবে।
- Existing account/history রাখতে পুরোনো `backend/data/nutrisight.sqlite3` backup রাখুন।
