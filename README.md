<div align="center">

# ⚡ FAST

### Efficient Federated Auxiliary Learning for Few-shot Medical Image Segmentation

**A little target data. A little help from the right clients. 🧩**

Jingyun Yang · Guoqing Zhang · Jingge Wang · Yang Li

[🏡 Homepage](https://yakult.beer) · [📧 Contact](mailto:yangjy20@tsinghua.org.cn)

</div>

---

## 👩🏻‍💻 Welcome!

This is the official repository for **FAST**, a **Federated Active and Sequential Targeted adaptation** framework for few-shot medical image segmentation.

FAST helps a newly joined client with limited labeled data adapt through selective collaboration with heterogeneous auxiliary clients.

### 🎯 Our goals

- **Make collaboration useful:** identify auxiliary clients that benefit the target.
- **Make adaptation efficient:** collaborate with one selected auxiliary client per round.
- **Make personalization reliable:** build a customized target model while maintaining global model reliability.

## 🪜 How FAST works

FAST actively selects auxiliary clients as target adaptation progresses, allowing collaboration to change with the target model’s needs.

1. **Select** an auxiliary client using a UCB-based multi-armed bandit strategy.
2. **Adapt** the target model with guidance from the selected auxiliary client.
3. **Repeat** selection and adaptation sequentially.
4. **Update** the global model once through gradient-aligned aggregation.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8d18e94e-7d1b-4b8d-8a28-f3c9bbdf69af" alt="Overview of the FAST framework, showing active auxiliary-client selection, sequential target adaptation, and one-time global aggregation." width="950">
</p>

## 🧀 Results

We evaluate FAST on **MRI and CT segmentation tasks** covering diverse anatomy and heterogeneous clients.

<p align="center">
  <img src="https://github.com/user-attachments/assets/907754aa-28d7-44fb-a0ec-4078336aef1d" alt="Experimental results comparing FAST with baseline methods for medical image segmentation." width="950">
</p>

## ⭐ Citation

If you find this work useful, please consider citing our paper. Thank you for supporting our research! 💛

*BibTeX details will be added when available.*

## 📬 Contact

Questions about FAST? Feel free to reach out!

**Jingyun Yang** · [yangjy20@tsinghua.org.cn](mailto:yangjy20@tsinghua.org.cn)

You can also visit my little corner of the internet: [yakult.beer](https://yakult.beer) 🍻
