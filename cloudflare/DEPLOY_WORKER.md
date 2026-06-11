# Deploy do Worker ANA Proxy

Proxy reverso Cloudflare entre o GitHub Actions (IP bloqueado pela ANA) e
`https://www.ana.gov.br/hidrowebservice`. Sem custo (free tier: 100k req/dia —
o sistema usa ~3.5k/dia).

## Arquitetura

```
GitHub Actions ──HTTPS──▶ Cloudflare Worker ──HTTPS──▶ www.ana.gov.br
(IP bloqueado)            (IP não bloqueado)           /hidrowebservice
```

- O `ana_collector.py` monta `CF_WORKER_URL + /hidrowebservice/<endpoint>`.
- O Worker repassa path, query e headers (`Identificador`, `Senha`,
  `Authorization`) para a ANA — **nenhuma credencial fica no Cloudflare**.
- A renovação automática de token (50 min) continua no coletor.

## Pré-requisitos

- Conta Cloudflare já ativa (a mesma do R2) ✅
- Node.js 18+ instalado (para o Wrangler CLI)

## Passos

### 1. Instalar o Wrangler

```powershell
npm install -g wrangler
```

### 2. Login no Cloudflare

```powershell
wrangler login
```

Abre o navegador — autorize com a mesma conta do R2.

### 3. Deploy (na raiz do projeto, onde está o wrangler.toml)

```powershell
wrangler deploy
```

> Não há secret para configurar no Worker — as credenciais da ANA viajam
> nos headers da requisição, vindas do `.env` local / GitHub Secrets.

### 4. Anotar a URL gerada

Formato: `https://ana-proxy-rs.SEU-USUARIO.workers.dev`

### 5. Adicionar no `.env` local

```
CF_WORKER_URL=https://ana-proxy-rs.SEU-USUARIO.workers.dev
```

### 6. Adicionar no GitHub Secrets

Repositório → Settings → Secrets and variables → Actions → New repository secret:

| Nome | Valor |
|---|---|
| `CF_WORKER_URL` | `https://ana-proxy-rs.SEU-USUARIO.workers.dev` |

(O workflow `collect_realtime.yml` já injeta `CF_WORKER_URL` no job `collect`.)

## Teste

### Teste 1 — autenticação via proxy (deve retornar `tokenautenticacao`)

```powershell
curl.exe -H "Identificador: $env:ANA_IDENTIFICADOR" `
         -H "Senha: $env:ANA_SENHA" `
         "https://ana-proxy-rs.SEU-USUARIO.workers.dev/hidrowebservice/EstacoesTelemetricas/OAUth/v1"
```

Resposta esperada: `{"status":"OK","items":{"tokenautenticacao":"eyJ..."}}`

### Teste 2 — coletor completo via proxy (local)

```powershell
$env:CF_WORKER_URL = "https://ana-proxy-rs.SEU-USUARIO.workers.dev"
python src/collectors/ana_collector.py
```

Logs esperados: `Token ANA obtido — válido por 50min.` e níveis dos rios > 0.

### Teste 3 — GitHub Actions

Actions → "Coleta Hidrometeorológica RS" → Run workflow → verificar no log do
step "Coletar níveis dos rios (ANA HidroWeb)" a mensagem `Token ANA obtido`.

## Troubleshooting

| Sintoma | Causa provável | Correção |
|---|---|---|
| 404 `Use /hidrowebservice/<endpoint>` | URL sem o prefixo `/hidrowebservice` | O coletor já adiciona; em testes manuais inclua o prefixo |
| 406 Not Acceptable | `Accept: application/json` forçado | O Worker repassa o `Accept: */*` do coletor — não altere |
| 401 no OAUth/v1 | `Identificador`/`Senha` errados ou ausentes | Conferir `.env` / GitHub Secrets |
| 522/timeout no Worker | ANA fora do ar (acontece) | O coletor tem retry exponencial (5x) |
