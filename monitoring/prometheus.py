"""Prometheus metrics for AURA-G services."""
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
except ImportError:
    raise SystemExit("pip install prometheus_client")

REQUESTS = Counter('aura_g_requests_total', 'Total requests',
                   ['endpoint', 'status'])
LATENCY = Histogram('aura_g_latency_seconds', 'Request latency',
                    ['endpoint'],
                    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30))
EVENTS = Counter('aura_g_events_total', 'Detected transcription events')
ACTIVE = Gauge('aura_g_active_requests', 'In-flight requests')
MODEL_LOAD = Gauge('aura_g_model_loaded', 'Model loaded flag', ['name'])

def instrument(app):
    """Attach Prometheus middleware to a FastAPI app."""
    from time import perf_counter
    from starlette.middleware.base import BaseHTTPMiddleware

    class MetricsMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            ACTIVE.inc()
            t0 = perf_counter()
            try:
                response = await call_next(request)
                status = str(response.status_code)
            except Exception:
                status = '500'
                raise
            finally:
                dt = perf_counter() - t0
                LATENCY.labels(endpoint=request.url.path).observe(dt)
                REQUESTS.labels(endpoint=request.url.path, status=status).inc()
                ACTIVE.dec()
            return response

    app.add_middleware(MetricsMiddleware)
    start_http_server(9090)
    print("Prometheus metrics on :9090/metrics")
