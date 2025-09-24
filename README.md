## Creacion del entorno

```
python3 -m venv project_env
```

## Acceso al entorno

```
source project_env/bin/activate
```
### Instalacion de ollama

Linux debian
```
curl -fsSL https://ollama.com/install.sh | sh
```

Linux centos

```
brew install ollama/tap/ollama
```

Para windows descargarlo e instalarlo

Verificacion de la instalacion

```
ollama --version
```

### Inatalar el modelo en ollama
```
ollama pull llama2:7b-chat
```