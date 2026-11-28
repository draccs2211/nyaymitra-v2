import { api, ApiError, errorMessage, setToken, setUnauthorizedHandler } from "../lib/api";

function mockFetch(status: number, body: unknown) {
  const fn = vi.fn().mockResolvedValue({ ok: status < 400, status, json: () => Promise.resolve(body) });
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => {
  vi.unstubAllGlobals();
  setToken(null);
});

describe("api client", () => {
  it("logs in with the OAuth2 form encoding", async () => {
    const fetch = mockFetch(200, { access_token: "t", token_type: "bearer", user_name: "A", user_role: "admin" });
    await api.login("a@b.in", "pw secret");
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe("/api/auth/login");
    expect(new Headers(init.headers).get("Content-Type")).toBe("application/x-www-form-urlencoded");
    expect(init.body).toBe("username=a%40b.in&password=pw+secret");
  });

  it("sends the bearer token and JSON body", async () => {
    setToken("abc");
    const fetch = mockFetch(200, {});
    await api.query({ query: "q", jurisdiction: "IN-UP" });
    const [, init] = fetch.mock.calls[0];
    const headers = new Headers(init.headers);
    expect(headers.get("Authorization")).toBe("Bearer abc");
    expect(headers.get("Content-Type")).toBe("application/json");
    expect(JSON.parse(init.body)).toEqual({ query: "q", jurisdiction: "IN-UP" });
  });

  it("turns errors into ApiError with the server's detail", async () => {
    mockFetch(429, { detail: "Daily AI usage limit reached for your firm." });
    await expect(api.cases()).rejects.toEqual(new ApiError(429, "Daily AI usage limit reached for your firm."));
  });

  it("signs out on 401 when a token was used", async () => {
    setToken("expired");
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    mockFetch(401, { detail: "Could not validate credentials" });
    await expect(api.me()).rejects.toBeInstanceOf(ApiError);
    expect(handler).toHaveBeenCalledOnce();
  });

  it("reads FastAPI validation errors", () => {
    expect(errorMessage({ detail: [{ msg: "password must be 8-72 bytes long" }] }, "x"))
      .toBe("password must be 8-72 bytes long");
    expect(errorMessage(null, "fallback")).toBe("fallback");
  });
});
