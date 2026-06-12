# Deploy do Cron Trigger (disparador externo do workflow)

O cron nativo do GitHub Actions é degradado em repositórios free (observado
em 11-12/06/2026: `*/10` executando a cada 2-4h e parando por horas).
Este Worker usa o **Cron Trigger do Cloudflare** (grátis, confiável) para
disparar o `workflow_dispatch` do `collect_realtime.yml` a cada 10 minutos.

```
Cloudflare Cron (*/10, confiável) ──▶ Worker climate-rs-cron
        └──POST /actions/workflows/collect_realtime.yml/dispatches──▶ GitHub
                                                └──▶ run do workflow (coleta)
```

## Passo 1 — Criar o Fine-grained PAT no GitHub (~2 min)

1. github.com → foto de perfil → **Settings** → **Developer settings**
   → **Fine-grained tokens** → **Generate new token**
2. Preencher:
   - **Token name**: `climate-rs-cron-trigger`
   - **Expiration**: 90 days (ou mais — anotar para renovar)
   - **Repository access**: *Only select repositories* → `climate-dashboard-rs`
   - **Permissions → Repository permissions → Actions**: **Read and write**
     (nenhuma outra permissão é necessária)
3. **Generate token** e **copiar** (formato `github_pat_...`) — só aparece uma vez

## Passo 2 — Configurar o secret no Worker (~1 min)

Na raiz do projeto:

```powershell
wrangler secret put GH_PAT -c wrangler.cron.toml
```

Colar o token quando solicitado. (O PAT fica criptografado no Cloudflare —
nunca no código nem no repo.)

> Se o wrangler reclamar que o Worker não existe ainda, rode primeiro o
> deploy do Passo 3 e depois o secret — ou aceite o prompt de criação.

## Passo 3 — Deploy (~30s)

```powershell
wrangler deploy -c wrangler.cron.toml
```

Saída esperada: `schedules: */10 * * * *`.

## Passo 4 — Verificar

1. **Imediato**: Cloudflare Dashboard → Workers & Pages → `climate-rs-cron`
   → aba *Logs* → aguardar o próximo múltiplo de 10 min → deve aparecer
   `workflow_dispatch OK`.
2. **GitHub**: Actions → "Coleta Hidrometeorológica RS" → novos runs a cada
   ~10 min (trigger aparece como `workflow_dispatch`).
3. **Supabase**: `live_*` com `updated_at` avançando a cada ciclo
   (lag < 30 min sustentado — critério da Etapa 1).

## Manutenção

| Item | Quando | Ação |
|---|---|---|
| PAT expira | conforme expiration | Gerar novo + `wrangler secret put GH_PAT -c wrangler.cron.toml` |
| Pausar coleta | se necessário | Cloudflare → Worker → Triggers → desabilitar cron (ou `crons = []` + redeploy) |
| Cron do GitHub | manter como redundância | O `schedule:` continua no workflow — se o GitHub disparar também, o `concurrency: collect-rs` evita sobreposição |

## Troubleshooting

| Sintoma | Causa | Correção |
|---|---|---|
| Log `HTTP 401` | PAT inválido/expirado | Regenerar PAT + secret |
| Log `HTTP 403` | PAT sem permissão Actions RW ou repo errado | Conferir escopo do token |
| Log `HTTP 404` | Nome do workflow/repo errado | Conferir constantes no .js |
| Runs duplicados | GitHub cron + Cloudflare juntos | Normal — `concurrency` cancela o excedente |
