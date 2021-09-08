### Knowledge Graph Augmented Political Perspective Detection in News Media
This repository serves as a code listing for the paper 'Knowledge Graph Augmented Political Perspective Detection in News Media'. Arxiv version of the paper is available at https://arxiv.org/abs/2108.03861. Work in progress.

#### Abstract
Identifying political perspective in news media has become an important task due to the rapid growth of political commentary and the increasingly polarized political ideologies. Previous approaches tend to focus on leveraging textual information and leave out the rich social and political context that helps individuals identify stances. To address this limitation, in this paper, we propose a perspective detection method that incorporates external knowledge of real-world politics. Specifically, we construct a political knowledge graph with 1,071 entities and 10,703 triples. We then construct heterogeneous information networks to represent news documents, which jointly model news text and external knowledge. Finally, we apply gated relational graph convolutional networks and conduct political perspective detection as graph-level classification. Extensive experiments demonstrate that our method consistently achieves the best performance and outperforms state-of-the-art methods by at least 5.49%. Ablation studies further bear out the necessity of external knowledge and the effectiveness of our graph-based approach.

#### Knowledge Graph
We construct a contemporary U.S. political knowledge graph and present it in /KG. It could serve as external knowledge for various tasks such as perspective detection and misinformation detection.


