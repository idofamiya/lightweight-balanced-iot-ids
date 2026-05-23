# GitHub Repository Setup Commands

## Option A: Using GitHub website

1. Go to https://github.com/new
2. Repository name: lightweight-balanced-iot-ids
3. Visibility: Public
4. Do not initialize with README, because this package already has one.
5. Click Create repository.

Then run these commands from inside this folder:

```bash
git init
git add .
git commit -m "Initial release of lightweight balanced IoT IDS experiments"
git branch -M main
git remote add origin https://github.com/idofamiya/lightweight-balanced-iot-ids.git
git push -u origin main
```

## Option B: Using GitHub CLI

```bash
gh repo create lightweight-balanced-iot-ids --public --source=. --remote=origin --push
```


