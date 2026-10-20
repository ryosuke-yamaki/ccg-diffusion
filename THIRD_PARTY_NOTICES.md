# Third-Party Notices

The MIT License ([`LICENSE`](LICENSE)) covers the code, documentation and data
written for this project. The upstreams below are the exceptions: ported or
adapted code keeps the license of its source, and third-party parser resources
and dataset contents keep their own terms. Each section names the upstream,
what this repository uses or ports from it and where, its license or terms, and
how to cite it. Licenses are as found at the pinned sources (checked on
2026-10-06). Model weights, evaluator weights and the official benchmark files
are downloaded by the setup scripts and are not redistributed here.

## Predicated Diffusion

- Upstream: <https://github.com/tksmatsubara/PredicatedDiffusion> at commit
  `af06f1be19dc6f0be468b271714c4904e623928e`.
- License: none published (no license file in the repository, none declared on
  GitHub). See [Code without a published license](#code-without-a-published-license).
- Used in: the Predicated Diffusion loss and the guided denoising loop that
  CCG Diffusion extends (`src/ccg_diffusion/guidance/predicated_diffusion_loss.py`,
  `guidance/attention.py`, `guidance/latent_update.py`,
  `backbones/sd14/pipeline.py`) and the Predicated Diffusion baseline with its
  guidance-target rules (`baselines/predicated_diffusion.py`).
- Paper: Sueyoshi and Matsubara, "Predicated Diffusion: Predicate Logic-Based
  Attention Guidance for Text-to-Image Diffusion Models", CVPR 2024.

```bibtex
@inproceedings{sueyoshi2024predicated,
    title = "Predicated Diffusion: Predicate Logic-Based Attention Guidance for Text-to-Image Diffusion Models",
    author = "Sueyoshi, Kota and Matsubara, Takashi",
    booktitle = "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
    year = "2024", pages = "8651--8660", doi = "10.1109/CVPR52733.2024.00826",
}
```

## SynGen

- Upstream: <https://github.com/RoyiRa/Linguistic-Binding-in-Diffusion-Models>
  at commit `50b8eab175dc9b31ce9c7c4b2eba28c817fc440e`.
- License: none published (no license file in the repository, none declared on
  GitHub). See [Code without a published license](#code-without-a-published-license).
- Used in: the SynGen baseline (`src/ccg_diffusion/baselines/syngen.py`, its
  Sana adaptation in `baselines/sana_plans.py` and `baselines/sana_generation.py`)
  and the noun rule of the Attend-and-Excite targets
  (`baselines/attend_and_excite.py`, from SynGen's `dvmp.py`).
- Paper: Rassin et al., "Linguistic Binding in Diffusion Models: Enhancing
  Attribute Correspondence through Attention Map Alignment", NeurIPS 2023.

```bibtex
@inproceedings{rassin2023linguistic,
    title = "Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment",
    author = "Rassin, Royi and Hirsch, Eran and Glickman, Daniel and Ravfogel, Shauli and Goldberg, Yoav and Chechik, Gal",
    booktitle = "Advances in Neural Information Processing Systems (NeurIPS)",
    volume = "36", year = "2023", pages = "3536--3559",
}
```

## Code without a published license

Neither Predicated Diffusion nor SynGen publishes a license (re-checked on
2026-10-08); their READMEs ask users to cite the papers. The files above are
distributed with this attribution so that CCG Diffusion and the compared
methods can be run and reproduced. The parts derived from these upstreams are
not covered by the MIT License of this repository, and this repository grants
no rights in them. The Predicated Diffusion implementation was itself built by
modifying Attend-and-Excite, so its Attend-and-Excite parts are under the MIT
License listed below. If you use these files, cite the Predicated Diffusion
and SynGen papers. The authors of either project can ask for the files to be
removed or replaced by opening an issue on this repository.

## Attend-and-Excite

- Upstream: <https://github.com/yuval-alaluf/Attend-and-Excite> at commit
  `163efdfd341bf3590df3c0c2b582935fbc8e8343`.
- License: MIT, Copyright (c) 2023 AttendAndExcite (text [below](#mit-license-text)).
- Used in: `src/ccg_diffusion/baselines/attend_and_excite.py` (objective,
  Gaussian smoothing, step schedule, refinement), and through Predicated
  Diffusion in the files listed there.
- Paper: Chefer et al., "Attend-and-Excite: Attention-Based Semantic Guidance
  for Text-to-Image Diffusion Models", ACM Transactions on Graphics 42(4), 2023.

```bibtex
@article{chefer2023attend,
    title = "Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models",
    author = "Chefer, Hila and Alaluf, Yuval and Vinker, Yael and Wolf, Lior and Cohen-Or, Daniel",
    journal = "ACM Transactions on Graphics", volume = "42", number = "4", articleno = "148",
    year = "2023", doi = "10.1145/3592116",
}
```

## Composable Diffusion

- Upstream: <https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch>
  at commit `77af82bbc73bbc987ca238bb9a3b0902ea0e12d0`.
- License: MIT, Copyright (c) 2022 Nan Liu and Shuang Li and Yilun Du (text
  [below](#mit-license-text)). The license file also appends the use-based
  restrictions of the Stable Diffusion model license for "the Model"; no model
  is distributed here.
- Used in: `src/ccg_diffusion/baselines/composable_diffusion.py` (component
  prompts, score composition).
- Paper: Liu et al., "Compositional Visual Generation with Composable Diffusion
  Models", ECCV 2022.

```bibtex
@inproceedings{liu2022compositional,
    title = "Compositional Visual Generation with Composable Diffusion Models",
    author = "Liu, Nan and Li, Shuang and Du, Yilun and Torralba, Antonio and Tenenbaum, Joshua B.",
    booktitle = "Proceedings of the European Conference on Computer Vision (ECCV)",
    year = "2022", pages = "423--439", doi = "10.1007/978-3-031-19790-1_26",
}
```

## StructureDiffusion

- Upstream: <https://github.com/weixi-feng/Structured-Diffusion-Guidance> at
  commit `773b2ba068c6cc60443d5d747a725413c8c9d46f`.
- License: MIT, Copyright (c) 2022 Weixi Feng (text [below](#mit-license-text)).
  The license file also appends the use-based restrictions of the Stable
  Diffusion model license for "the Model"; no model is distributed here.
- Used in: `src/ccg_diffusion/baselines/structure_diffusion.py` (noun-phrase
  extraction and alignment, structured cross-attention).
- Paper: Feng et al., "Training-Free Structured Diffusion Guidance for
  Compositional Text-to-Image Synthesis", ICLR 2023.

```bibtex
@inproceedings{feng2023trainingfree,
    title = "Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis",
    author = "Feng, Weixi and He, Xuehai and Fu, Tsu-Jui and Jampani, Varun and Akula, Arjun and Narayana, Pradyumna and Basu, Sugato and Wang, Xin Eric and Wang, William Yang",
    booktitle = "The Eleventh International Conference on Learning Representations (ICLR)",
    year = "2023", url = "https://openreview.net/forum?id=PUIqjT4rzq7",
}
```

## T2I-CompBench++

- Upstream: <https://github.com/Karine-Huang/T2I-CompBench> at commit
  `1b7094991a57f3c22abdd4f6e8ba6c1a15517073`.
- License: MIT, Copyright (c) 2023 HKU (text [below](#mit-license-text)).
- Used in: `scripts/t2i_compbench/official_reference/generate.py` (port of
  `GORS_finetune/inference_eval.py`); the official test prompts in
  `dataset/derived/t2i-compbench/t2i-compbench-full-v1/` and, as parse inputs,
  in `data/t2i_compbench/`, `data/compute_cost/` and `tests/fixtures/`;
  `docker/t2i_compbench/` (requirements derived from the official
  `requirements.txt`). The official evaluators run unmodified from a clone
  installed by `scripts/setup/setup_t2i_compbench.py`.
- Paper: Huang et al., "T2I-CompBench++: An Enhanced and Comprehensive
  Benchmark for Compositional Text-to-Image Generation", IEEE TPAMI 47(5), 2025.

```bibtex
@article{huang2025t2icompbenchpp,
    title = "{T2I-CompBench++}: An Enhanced and Comprehensive Benchmark for Compositional Text-to-Image Generation",
    author = "Huang, Kaiyi and Duan, Chengqi and Sun, Kaiyue and Xie, Enze and Li, Zhenguo and Liu, Xihui",
    journal = "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    volume = "47", number = "5", pages = "3563--3579", year = "2025", doi = "10.1109/TPAMI.2025.3531907",
}
```

## prompt-to-prompt

- Upstream: <https://github.com/google/prompt-to-prompt>.
- License: Apache License 2.0 (<https://www.apache.org/licenses/LICENSE-2.0>).
- Used in: the attention-store pattern of the cross-attention capture
  (`AttentionStore` in `src/ccg_diffusion/guidance/attention.py`), via
  Predicated Diffusion and Attend-and-Excite.
- Paper: Hertz et al., "Prompt-to-Prompt Image Editing with Cross-Attention
  Control", ICLR 2023.

## Diffusers

- Upstream: <https://github.com/huggingface/diffusers>, version 0.39.0
  (installed dependency).
- License: Apache License 2.0 (<https://www.apache.org/licenses/LICENSE-2.0>).
- Used in: the attention processors and the Sana denoising loop
  (`src/ccg_diffusion/backbones/sana/guidance.py`) follow the structure of the
  diffusers 0.39.0 implementations.

## Hol-CCG and the C&C resources

- Upstream: `holccg` 1.0.0, <https://github.com/ryosuke-yamaki/hol-ccg> at
  commit `0213ca85d2397c5a4fb2c03bc1b20cf1a6283d26` (installed dependency), and
  the checkpoint `ryosuke-yamaki/hol-ccg-roberta-large` at revision
  `1888cd6343d8df52cf871e1a86924f500a6d7b6b` (base model
  `FacebookAI/roberta-large`, MIT; downloaded, not redistributed).
- License: the `holccg` code and the checkpoint are MIT, Copyright (c) 2023
  Ryosuke Yamaki. The C&C `generate` port and the C&C grammar resources bundled
  in `holccg` (`markedup`, `trNP`, `trPP`, `trAP`, `trVP_to`) keep the C&C
  tools licence (non-commercial use) and the Java C&C BSD 2-Clause licence, as
  stated in the `holccg` notices (`License-Expression: MIT AND
  LicenseRef-CandC-Non-Commercial`).
- Used in: Stage 1 parsing and Supertag-Constrained Parsing
  (`src/ccg_diffusion/parsing/holccg.py`), Stage 2 marked dependencies
  (`parsing/dependencies.py`, `scripts/ccg/generate_dependencies.py`).
  [`configs/parsing/candc_markedup_additions`](configs/parsing/candc_markedup_additions)
  adds categories to the C&C markedup file and is used under the same terms as
  the C&C resources. The marked dependencies in `data/t2i_compbench/` are output
  of the C&C `generate` port.
- Terms: the C&C tools licence (Universities of Edinburgh, Oxford and Sydney;
  James R. Curran) permits use for education and research only, requires
  copies to keep the copyright notice, and requires research that uses the
  tools to cite Clark and Curran (2004). Stage 2 is part of every CCG Diffusion
  run, so these terms apply to running the method.

```bibtex
@inproceedings{yamaki2023holographic,
    title = "Holographic {CCG} Parsing",
    author = "Yamaki, Ryosuke and Taniguchi, Tadahiro and Mochihashi, Daichi",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2023", pages = "262--276", url = "https://aclanthology.org/2023.acl-long.15",
}

@inproceedings{clark2004parsing,
    title = "Parsing the {WSJ} Using {CCG} and Log-Linear Models",
    author = "Clark, Stephen and Curran, James R.",
    booktitle = "Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics (ACL)",
    year = "2004", pages = "103--110", url = "https://aclanthology.org/P04-1014",
}
```

## TAB (Text-to-image Ambiguity Benchmark)

- Upstream: <https://github.com/amazon-science/tab-dataset> at commit
  `2e9d70c3c44037ef7e6a2c96bf8efb31bd6aed0e`.
- License: CC BY-SA 4.0 (<https://creativecommons.org/licenses/by-sa/4.0/>).
- Used in: `dataset/derived/tab/tab-ppvp-disambiguation-v1/` (items copied
  verbatim) and the derived bundle `data/tab_pp/`, both distributed under
  CC BY-SA 4.0; the TAB sentences quoted in `tests/` (fixtures and test cases)
  follow the same license, while the rest of those files is under the MIT
  License. The official CSV files are downloaded by
  `scripts/data/prepare_tab_pp_benchmark.py` and
  `scripts/data/prepare_tab_vp_benchmark.py` and are not redistributed. Run
  outputs that you generate quote TAB text as well.
- Paper: Mehrabi et al., "Resolving Ambiguities in Text-to-Image Generative
  Models", ACL 2023.

```bibtex
@inproceedings{mehrabi2023resolving,
    title = "Resolving Ambiguities in Text-to-Image Generative Models",
    author = "Mehrabi, Ninareh and Goyal, Palash and Verma, Apurv and Dhamala, Jwala and Kumar, Varun and Hu, Qian and Chang, Kai-Wei and Zemel, Richard and Galstyan, Aram and Gupta, Rahul",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2023", pages = "14367--14388", url = "https://aclanthology.org/2023.acl-long.804",
}
```

## Models and services

Downloaded or called at run time; none of them is redistributed here.

| Model or service | Role | License or terms |
|---|---|---|
| `CompVis/stable-diffusion-v1-4` @ `133a221b8aa7292a167afc5127cb63fb5005638b` | backbone (Stable Diffusion v1.4) | CreativeML OpenRAIL-M |
| `Efficient-Large-Model/Sana_600M_512px_diffusers` @ `83d7a190bfd1fd070570a793d2dab5c7a3231b9d` | backbone (Sana 0.6B) | Apache-2.0 (license of the model repository); its text encoder is a Gemma 2 model, provided under the Gemma Terms of Use |
| `ryosuke-yamaki/hol-ccg-roberta-large` @ `1888cd6343d8df52cf871e1a86924f500a6d7b6b` | CCG parser checkpoint | MIT |
| spaCy `en_core_web_sm` 3.8.0, `en_core_web_trf` 3.6.1 | possession lemmas and Predicated Diffusion targets; baseline preprocessing | MIT |
| Stanza 1.4.2 English models | StructureDiffusion noun phrases | Stanza's license (Apache-2.0 library) |
| T2I-CompBench++ evaluator weights (BLIP, UniDet, MiDaS, CLIP) | official metrics | the licenses of their sources; downloaded by `scripts/setup/setup_t2i_compbench.py` |
| GPT-6 Astra (OpenAI API) | TAB PP judge | OpenAI terms of use |
| Qwen3.8-Flash-Next (`nvidia/Qwen3.8-Flash-Next-NVFP4`, served with vLLM) | TAB PP judge | Qwen community license of the base model; NVIDIA Open Model License of the NVFP4 checkpoint; vLLM is Apache-2.0 |

## MIT License text

The MIT License text that applies to the MIT components above, with the
copyright lines given in their sections:

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
