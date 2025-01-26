---
marp: true
theme: default
paginate: true
header: 'Chest X-Ray Classification System'
footer: 'Confidential & Proprietary - 2024'
style: |
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  .small-text {
    font-size: 0.8em;
  }
---

<!-- _class: lead -->
# Chest X-Ray Classification System
## Executive Summary

![bg right:40% 80%](https://via.placeholder.com/800x600?text=X-Ray+AI)

---

# Business Value

- **Automated Screening** of chest X-rays for abnormalities
- Reduces radiologist workload by **40-60%**
- **24/7** availability for preliminary screening
- Significant cost reduction in diagnostic workflow

---

# Key Performance Metrics


**Micro Metrics**
- Precision: 20%
- Recall: 58%
- F1-Score: 29.5%

**Operational Metrics**
- Processing time: <2 seconds/image
- System uptime: 99.9%
- Cost per analysis: $0.02

---

# ROI Analysis


**Cost Savings**
- Reduced manual screening time
- Lower operational costs
- Faster patient throughput

**Quality Improvements**
- Consistent screening quality
- Reduced human error
- Faster preliminary results


---


# Performance Metrics

![height:400px](images/auc_by_condition_resnet.png)

---

# Data Pipeline

![height:700px](images/mermaid-diagram-2025-01-26-140527.png)

---



# Infrastructure

```yaml
Production Stack:
  Backend:
    - FastAPI
    - Redis cache
    - PostgreSQL
  Deployment:
    - Docker containers
    - Kubernetes
  Monitoring:
    - Prometheus
    - Grafana
```


---

# Monitoring System


**Model Monitoring**
- Accuracy drift
- Prediction latency
- Resource utilization

**System Health**
- API endpoints
- Database connections
- Cache hit ratio


---

# Security Measures


**Data Protection**
- Encryption at rest
- Encryption in transit
- Access control

**Model Security**
- Input validation
- Rate limiting
- Audit logging

---

# Future Roadmap

**Short-term (Q2 2024)**
- Batch processing
- Real-time monitoring
- Collect more data
- Larger images

---

# Questions & Discussion


---
