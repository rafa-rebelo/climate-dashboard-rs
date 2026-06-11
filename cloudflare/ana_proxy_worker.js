/**
 * ANA HidroWeb Proxy — Cloudflare Worker
 *
 * Proxy reverso transparente entre o GitHub Actions (IP bloqueado pela ANA)
 * e https://www.ana.gov.br/hidrowebservice.
 *
 * Desenho: pass-through de path + query + headers de autenticação.
 * A autenticação (Identificador/Senha → token Bearer) é feita pelo próprio
 * ana_collector.py — o Worker NÃO armazena credenciais da ANA.
 *
 * Fluxo:
 *   GET https://ana-proxy-rs.X.workers.dev/hidrowebservice/EstacoesTelemetricas/OAUth/v1
 *     → GET https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas/OAUth/v1
 *
 * Headers repassados: Identificador, Senha, Authorization, Accept, Content-Type.
 * Accept NÃO é forçado para application/json (a ANA retorna 406 em vários
 * endpoints quando isso acontece — ver comentário em ana_collector.py).
 */

const ANA_ORIGIN = "https://www.ana.gov.br";

// Headers que o coletor envia e devem chegar à ANA.
const FORWARD_HEADERS = [
  "identificador",
  "senha",
  "authorization",
  "accept",
  "content-type",
];

export default {
  async fetch(request) {
    const url = new URL(request.url);

    // Aceita apenas chamadas ao serviço HidroWeb — evita uso do Worker
    // como proxy aberto para qualquer path da ana.gov.br.
    if (!url.pathname.startsWith("/hidrowebservice/")) {
      return new Response(
        JSON.stringify({ error: "Use /hidrowebservice/<endpoint>" }),
        { status: 404, headers: { "Content-Type": "application/json" } },
      );
    }

    const target = ANA_ORIGIN + url.pathname + url.search;

    const headers = new Headers();
    for (const name of FORWARD_HEADERS) {
      const value = request.headers.get(name);
      if (value) headers.set(name, value);
    }
    // User-Agent de browser — a ANA recusa alguns User-Agents programáticos.
    headers.set(
      "User-Agent",
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    );

    const upstream = await fetch(target, {
      method: request.method,
      headers,
      body: ["GET", "HEAD"].includes(request.method) ? undefined : request.body,
    });

    // Repassa status e corpo originais; só adiciona CORS.
    const respHeaders = new Headers(upstream.headers);
    respHeaders.set("Access-Control-Allow-Origin", "*");

    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: respHeaders,
    });
  },
};
