# NutriSight UI reference match

This frontend was adjusted to follow the supplied 12-screen NutriSight reference as closely as possible while keeping the existing Expo/React Native app functional.

Matched reference flow and styling:
- Splash / brand screen
- Smart Nutrition intro
- Login
- Sign up
- Profile
- Scan Food
- Analyzing Food
- Detected Items
- Nutrition Results
- Food History
- Settings
- Nutrition Insights

Important functional choices:
- Sign-up now uses the same visible fields as the reference: Full Name, Email, Password.
- Existing API authentication, image picking, analysis, history, profile storage, nutrition results, and theme logic remain connected.
- The backend connection control is kept under Settings > About so the main Settings screen stays visually close to the reference.
- Dynamic user/model data is shown instead of hard-coded names, calories, dates, and foods from the mockup.
