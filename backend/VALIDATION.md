# Validation status

Validated in this build environment:

- `python -m pytest tests -q` → **5 passed**.
- Signup, login, session expiry, profile save, onboarding skip, private history, raw image upload contract, password change and CORS are covered by API tests.
- Supplied checkpoint contracts were inspected successfully with `torch.load`:
  - YOLO checkpoint file present.
  - ViT classifier head has 137 outputs and matches the supplied 137-entry class mapping.
  - Nutrition checkpoint contains the expected targets: calories, mass, fat, carb and protein.
- All frontend TypeScript/TSX files were parsed with the TypeScript compiler API with **0 syntax errors**.
- Legacy demo imports of `expo-image` / `expo-symbols` were removed from the final frontend source.

Not executed in this container:

- Full real-model inference, because `timm` and `ultralytics` are not preinstalled here and package download is unavailable in the build environment.
- Full Expo Android/Web bundle, because npm dependency download is unavailable in the build environment.
- Physical phone camera testing.

Run `setup.bat` on your project machine, then the included test/model-contract commands for local verification.
