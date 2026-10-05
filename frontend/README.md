# NutriSight Frontend v4 — Reference UI Build

Expo/React Native frontend rebuilt to closely reproduce the supplied NutriSight mobile UI reference.

## Implemented reference screens

1. NutriSight branded splash screen
2. “Smart Nutrition Starts Here” onboarding screen
3. Welcome Back / Sign In
4. Create Account / Sign Up
5. First-login nutrition profile setup with Skip
6. Home dashboard in the same green/cream visual system
7. My Profile summary + editable profile
8. Scan Food: camera and gallery cards
9. Analyzing Food progress screen
10. Detected Items with real backend detection boxes
11. Nutrition Results
12. My Food History
13. Settings
14. Nutrition Insights with macro donut

The supplied reference is stored at `docs/user-ui-reference.png`.

## Design system

- Warm ivory background and clean white cards
- Deep green primary actions with soft botanical green accents
- Quicksand typography
- Leaf/fork NutriSight logo drawn as vector UI
- Rounded cards, subtle shadows, compact mobile spacing
- Reference-style bottom navigation with raised center Scan action
- Custom vector food/phone, meal and analyzing illustrations
- Light/dark support retained

## Functional integrations retained

- Email/password signup and login
- First-login profile setup and Skip
- Camera/gallery image selection
- Native raw-file upload to `/api/analyze/raw`
- YOLO → ViT fallback → nutrition backend flow
- Detection boxes using backend coordinates
- Nutrition result data
- Saved history
- Profile editing
- Password change
- Backend address + health/model status in Settings
- Logout and account deletion

Google/Apple buttons are visual placeholders because the supplied backend does not contain OAuth endpoints. They show a clear message instead of pretending to authenticate.

## Setup

Run once:

```bat
setup.bat
```

Then:

```bat
start.bat
```

For a physical phone, use the PC's LAN address in **Settings → Backend Connection**, for example:

```text
http://192.168.0.25:8000
```

Do not use `127.0.0.1` from a physical phone.

## Native scan upload

Native scans intentionally send an Expo FileSystem `File` as the request body rather than multipart FormData. This avoids the earlier `Unsupported FormDataPart implementation` issue.
