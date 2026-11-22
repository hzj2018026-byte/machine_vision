# 识矿乐园 · 矿石识别科普 Web

基于《识果乐园》需求文档改编的矿石主题前后端项目：

- 后端：FastAPI + SQLAlchemy + MySQL（也支持 SQLite 演示）
- 前端：Vue 3 + Vite + vue-router + Axios
- 模型：直接复用当前项目的 ResNet18 checkpoint
  `models/ore_classification_model.pth`
- 5 类矿石：方解石、石英、赤铁矿、长石、黄铁矿

完整开发说明见项目根目录的 [《识矿乐园项目开发说明.md》](../识矿乐园项目开发说明.md)。

## 快速启动

```powershell
# 1. 配置数据库（后端启动时会自动 CREATE DATABASE IF NOT EXISTS）
copy .\backend\.env.example .\backend\.env
# 修改 backend/.env 里的 DB_PASSWORD
# 不方便装 MySQL 时，可改成：DATABASE_URL=sqlite:///./ore_recognition.db

# 2. 启动后端
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000

# 3. 另开终端启动前端
cd frontend
npm install
npm run dev
```

访问 <http://localhost:5173>，管理员账号 `admin / 123456`。

## 一键启动脚本

| 脚本 | 用途 |
| --- | --- |
| `start_all.bat` | MySQL 后端 + 前端 |
| `start_all_sqlite.bat` | SQLite 后端 + 前端（不需要 MySQL，推荐先跑这个） |
| `start_backend.bat` | 只启动 MySQL 后端 |
| `start_backend_sqlite.bat` | 只启动 SQLite 后端 |
| `start_frontend.bat` | 只启动前端 |

> 使用 MySQL 前需要先创建 `ore_kids` 数据库：
> `Get-Content .\backend\init_db.sql | mysql -u root -p`

## 健康检查

```powershell
curl.exe http://127.0.0.1:8000/api/health
```

`model_ready` 为 `true` 才能正常识别。后端会自动读取 checkpoint 里的
`class_names` 和 `img_size`，兼容旧的 160×160 模型和新的 224×224 模型。
