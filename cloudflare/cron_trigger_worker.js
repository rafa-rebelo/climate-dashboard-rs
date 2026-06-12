/**
 * Cron Trigger — disparador externo do workflow de coleta
 *
 * O cron nativo do GitHub Actions é degradado em repos free (observado:
 * agenda de 10 min executando a cada 2-4h, às vezes parando por horas).
 * Este Worker usa o Cron Trigger do Cloudflare (confiável, grátis) para
 * disparar o workflow_dispatch a cada 10 minutos via API do GitHub.
 *
 * Secret necessário (NUNCA hardcoded):
 *   GH_PAT — Fine-grained PAT com permissão Actions: Read and write
 *            no repo climate-dashboard-rs.
 *   Configurar com: wrangler secret put GH_PAT -c wrangler.cron.toml
 *
 * Agendamento: definido em wrangler.cron.toml ([triggers] crons).
 */

const OWNER    = "rafa-rebelo";
const REPO     = "climate-dashboard-rs";
const WORKFLOW = "collect_realtime.yml";
const REF      = "main";

export default {
  async scheduled(event, env, ctx) {
    const url =
      `https://api.github.com/repos/${OWNER}/${REPO}` +
      `/actions/workflows/${WORKFLOW}/dispatches`;

    const resp = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${env.GH_PAT}`,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "climate-rs-cron-trigger",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ref: REF }),
    });

    // 204 No Content = disparo aceito pelo GitHub
    if (resp.status !== 204) {
      const body = await resp.text();
      console.error(`workflow_dispatch falhou: HTTP ${resp.status} — ${body.slice(0, 300)}`);
    } else {
      console.log(`workflow_dispatch OK (${event.cron} @ ${new Date(event.scheduledTime).toISOString()})`);
    }
  },
};
