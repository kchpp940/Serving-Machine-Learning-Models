# 运行时配置环境变量参考

> 单一真相来源：`car_pricing.config.export_env_schema()`
> 本文件由 `scripts/generate_config_artifacts.py docs` 自动生成，请勿手动修改

## 概览

共 **13** 个环境变量，按类别分组：

- **API server configuration**: 2 个变量
- **Model file locations**: 6 个变量
- **Training data locations**: 2 个变量
- **Client configuration**: 2 个变量
- **BentoML configuration**: 1 个变量

## 详细说明


### API server configuration

| 变量名 | 默认值 | 类型 | 敏感 | 必需 | 说明 |
|--------|--------|------|------|------|------|
| `API_HOST` | `0.0.0.0` | str | ❌ | ✅ | API 监听地址 |
| `API_PORT` | `8000` | int | ❌ | ✅ | API 监听端口（Heroku 会自动注入 PORT 环境变量，启动时会映射到此变量） |


### Model file locations

| 变量名 | 默认值 | 类型 | 敏感 | 必需 | 说明 |
|--------|--------|------|------|------|------|
| `MODEL_DIR` | *<auto-resolved>* | str | ❌ | ✅ | 模型文件目录。未设置时按优先级解析：./models → ./shared_models → <project_root>/shared_models → <project_root>/models |
| `MODEL_FILENAME` | `sklearn_gbr.pkl` | str | ❌ | ✅ | 模型文件名 |
| `MODEL_METADATA_FILENAME` | `model_metadata.json` | str | ❌ | ✅ | 模型元数据文件名 |
| `MODEL_STATUS_FILENAME` | `model_status.json` | str | ❌ | ✅ | 模型状态文件名 |
| `SCHEMA_SNAPSHOT_FILENAME` | `schema_snapshot.json` | str | ❌ | ✅ | Schema snapshot 文件名（训练时由 lineage.persist() 写入完整特征/编码器快照） |
| `LINEAGE_FILENAME` | `model_lineage.json` | str | ❌ | ✅ | Model lineage 文件名（训练时由 lineage.persist() 写入完整血缘记录） |


### Training data locations

| 变量名 | 默认值 | 类型 | 敏感 | 必需 | 说明 |
|--------|--------|------|------|------|------|
| `DATA_CSV_FILENAME` | `cars.csv` | str | ❌ | ✅ | 训练数据 CSV 文件名 |
| `DATA_DIR` | *<auto-resolved>* | str | ❌ | ✅ | 训练数据目录。未设置时按优先级解析：./Data → <project_root>/Data |


### Client configuration

| 变量名 | 默认值 | 类型 | 敏感 | 必需 | 说明 |
|--------|--------|------|------|------|------|
| `API_BASE_URL` | `http://localhost:8000` | str | ❌ | ✅ | API 基础 URL（供 Streamlit 等客户端调用） |
| `REQUEST_TIMEOUT` | `10` | int | ❌ | ✅ | 请求超时时间（秒） |


### BentoML configuration

| 变量名 | 默认值 | 类型 | 敏感 | 必需 | 说明 |
|--------|--------|------|------|------|------|
| `BENTOML_MODEL_TAG` | `gbr:latest` | str | ❌ | ✅ | BentoML 模型标签 |

## 部署目标默认值差异

不同部署平台对路径等变量有不同的默认值约定：

| 变量名 | Schema本地默认 | Docker/Heroku | Shell 脚本 | Vercel |
|--------|-----------------|-----------------|-----------------|-----------------|
| `API_HOST` | `0.0.0.0` | 同左 | 同左 | 同左 |
| `API_PORT` | `8000` | 同左 | 同左 | 同左 |
| `MODEL_DIR` | `<auto-resolved>` | `/app/shared_models` | `$PROJECT_ROOT/shared_models` | 同左 |
| `MODEL_FILENAME` | `sklearn_gbr.pkl` | 同左 | 同左 | 同左 |
| `MODEL_METADATA_FILENAME` | `model_metadata.json` | 同左 | 同左 | 同左 |
| `MODEL_STATUS_FILENAME` | `model_status.json` | 同左 | 同左 | 同左 |
| `SCHEMA_SNAPSHOT_FILENAME` | `schema_snapshot.json` | 同左 | 同左 | 同左 |
| `LINEAGE_FILENAME` | `model_lineage.json` | 同左 | 同左 | 同左 |
| `DATA_CSV_FILENAME` | `cars.csv` | 同左 | 同左 | 同左 |
| `DATA_DIR` | `<auto-resolved>` | `/app/Data` | `$PROJECT_ROOT/Data` | 同左 |
| `API_BASE_URL` | `http://localhost:8000` | `http://localhost:${API_PORT}` | `http://localhost:$API_PORT` | 同左 |
| `REQUEST_TIMEOUT` | `10` | 同左 | 同左 | 同左 |
| `BENTOML_MODEL_TAG` | `gbr:latest` | 同左 | 同左 | 同左 |

## 如何新增/修改配置

1. 编辑 `car_pricing/config.py`，在 `_build_env_var_registry()` 中添加/修改变量的 EnvVarMeta 定义
2. 如果需要全局 `_DEFAULT_*` 常量，在文件顶部添加
3. 如果变量在不同部署目标有不同默认值，直接在该 EnvVarMeta 的 `deployment_defaults` 字段中定义
4. 运行以下命令自动更新所有配置产物：

```bash
# 从 schema 重新生成所有配置产物（推荐）
python scripts/generate_config_artifacts.py

# 仅更新部署文件（.env.example + Dockerfile/Procfile/vercel.json/heroku.yml 标记段）
python scripts/generate_config_artifacts.py deploy

# 仅预览变更，不写入文件
python scripts/generate_config_artifacts.py dry-run

# 验证一致性
python tests/test_config_consistency.py
```

