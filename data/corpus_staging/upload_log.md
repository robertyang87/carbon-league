# Watchlist corpora warehousing — upload log
Source: DGX /home/arm/junyang/carbon-repo/data/{k4v3, hakimi3pro, fable6}_corpus (pulled 19:14-19:23 CST 2026-11-13) + /tmp/{label}_matches.json
Staged: ~/workspace/carbon-live/corpus_staging/ — local verify: counts 64/64/75, bytes 15583984/15762047/18153207, all gunzip clean, id-set == matches ids, md5 of all 206 files == DGX.
Target: robertyang87/carbon-league, branch main, paths data/{label}_corpus/<file>.json.gz

## matches.json decision (2026-11-13 ~19:35)
NOT uploaded. Repo data/{label}_matches.json is NEWER than the DGX 19:14 copies and a superset in all three cases:
- k4v3: repo 65 vs mine 64 (repo extra 0f680a5506e982d26d6c235f850289b7, created 19:20:07)
- hakimi3pro: repo 64 vs mine 64, identical id sets (bytes differ only by formatting)
- fable6: repo 76 vs mine 75 (repo extra 255dc1d8850905e907a40b03c457d7c3, created 19:25:51)
Uploading the 19:14 copies would REGRESS k4v3/fable6 manifests. DGX /tmp copies still 19:14 (not re-pulled).
Gap flagged: after this upload the corpus will hold 64/64/75 replays while repo manifests list 65/64/76 games — 2 replays (k4v3 0f680a55…, fable6 255dc1d8…) missing until a DGX top-up pull.

## uploads
| path | sha | bytes | time CST | note |
| data/k4v3_corpus/01b1d5ed7b7722e43da8384d18427b90-1.json.gz | b22f0e4bfad3c094534e3d2499612ff8e06e4f28 | 250044 | 11:33:10 |
| data/k4v3_corpus/0b45c91312d82628637ead73edce8779-1.json.gz | 0e8163e4b11d53e83bc36f7bc4053e65bf48cf06 | 247673 | 11:33:12 |
| data/k4v3_corpus/0ed2fdb5ad9e5fb262c876e36b64aff3-1.json.gz | 1ebc65d01653f87d75857295feff908c56eacc59 | 238793 | 11:33:14 |
| data/k4v3_corpus/14e6fb0d2fcb5f3471fd343360ea9253-1.json.gz | da52e3424eef0ecb12845f0420adde32bd20bdd3 | 241103 | 11:33:16 |
| data/k4v3_corpus/16540b6667cff18458bc0bbd4aa896a0-1.json.gz | be1217ad34cbe5ff33e529981745842a6ed27fab | 246627 | 11:33:18 |
| data/k4v3_corpus/18077e6fec29f70b6b0141da7b6ee3c9-1.json.gz | 963af296cac889067d657d8021df2d248003b171 | 226998 | 11:33:20 |
| data/k4v3_corpus/19277e72fb44f6010bf417530be02229-1.json.gz | c571d420951e07318f33199e215eb744570b59a3 | 232767 | 11:33:22 |
| data/k4v3_corpus/19595c6f1ef43cf50373e0cc2dd5ca7e-1.json.gz | 142969fb129ae8f2710e53c6f4530a5a9ce108ef | 237412 | 11:33:24 |
| data/k4v3_corpus/1f02972ce85cf65df9156dba0a920e7e-1.json.gz | d1d2e33e3ff155473ea27dab1d89b496c85304a3 | 249045 | 11:33:26 |
| data/k4v3_corpus/201870b05a15208638602e18cf9373d6-1.json.gz | 8644520953ca3c6dab4adb6979bdf217da98f63d | 242840 | 11:33:29 |
| data/k4v3_corpus/274c2752585fdb6673ecaeeeb3908c9a-1.json.gz | 09fec4a8a049b6c96fad59f48f6b8d5623574285 | 240472 | 11:33:30 |
| data/k4v3_corpus/2e08cb5bbc78f08b185731251f921a7b-1.json.gz | 8739edc1b594bf97c64b1718d3f0159e2c56f378 | 238770 | 11:33:32 |
| data/k4v3_corpus/367d063490d1edd8ecbb6e2aab0942c2-1.json.gz | e03f7658d9a9e94bfb316d1af3d661ee2be86afe | 235388 | 11:33:34 |
| data/k4v3_corpus/3b2d12994a75c746fc26eeaa4c04b5b2-1.json.gz | 866dfacae0f6634d13445b49c4ee7a531ebb4c92 | 252909 | 11:33:36 |
| data/k4v3_corpus/40c3254221e93bfa25cac42a0f44a770-1.json.gz | 624f2230559531f4a0b0e0e839426b9461ade446 | 239912 | 11:33:38 |
| data/k4v3_corpus/43c5425d588e8e999a8457af2ac8eb69-1.json.gz | 4820aab5e6e497a72aa2916947d8619c2229fee0 | 237478 | 11:33:40 |
| data/k4v3_corpus/49780405724c551096625488bc309a7b-1.json.gz | 487e45cf87fb05347b15d74da555c7ddce48ea1f | 238777 | 11:33:42 |
| data/k4v3_corpus/4a4c1ab20669310cc8e16de3a4d1d97d-1.json.gz | 2ed25a6194ed1dff73f2b2350ae4e307e9ab02ac | 245190 | 11:33:44 |
| data/k4v3_corpus/5390fb824dab2c072befed16240b6b15-1.json.gz | 3bfbd586dc2587fc01d46baa53f6a849e6374a50 | 252954 | 11:33:46 |
| data/k4v3_corpus/57daf2e4ed4611fb9f0ec02ea57f83fc-1.json.gz | 335b24de7e50d0ae296cf1b157d395e4c79386b8 | 242146 | 11:33:48 |
| data/k4v3_corpus/6103d2c3f154b957b2e268fb558d2b71-1.json.gz | 2ca3ac244aaaf7fc8315027a21c5a025f05c53d2 | 252023 | 11:33:51 |
| data/k4v3_corpus/6accf36e08d91300e3cd6c92e7bbd346-1.json.gz | 9c0b7a66c515999abc0a664ed94c94efff0281a8 | 243380 | 11:33:53 |
| data/k4v3_corpus/6c94ddda94959f90fe35b936becb71fa-1.json.gz | bba3130573215bde9c0c316ca90f90e6072e6be3 | 236081 | 11:33:55 |
| data/k4v3_corpus/6df6612764b6b783de0d7baa9d1b5d0e-1.json.gz | 8878ec62b4d2a4e0ef8f923b9a25001f26268360 | 243697 | 11:33:57 |
| data/k4v3_corpus/6f2a35181ec041c75d8ad0b81b473e14-1.json.gz | 5caae937b881d3484f9fd811690e7da6e878fd13 | 235683 | 11:34:00 |
| data/k4v3_corpus/77afac5ec98ad67d04826915d1a18812-1.json.gz | 688b4d84389d538fd51ba4118829d7d43de8068e | 243077 | 11:34:02 |
| data/k4v3_corpus/798855af569b47c271284a6740f5114a-1.json.gz | 0df3c0901b535a9670a9a1d8f9bd93782582c7f4 | 253776 | 11:34:04 |
| data/k4v3_corpus/79c711f34673f4fd93c2007537a97bb1-1.json.gz | 1019040ef5bc61ee9e6414f08bc7c17cad6c1289 | 238955 | 11:34:06 |
| data/k4v3_corpus/7b316aad7b749fe50fc585a99e6c1ed7-1.json.gz | 0d2501b28e9fa4efec94dfa6f2742614cbce409a | 237474 | 11:34:08 |
| data/k4v3_corpus/7d5f954c4f1e3c776f32725f51a2bc4b-1.json.gz | f8c7121a9ce3a3b600178d2e71038d0210966aa7 | 241296 | 11:34:10 |
| data/k4v3_corpus/856b1d5ec4be53eea879cf9e7ef1d238-1.json.gz | d24e77b8eee52240f04078dbf6dcee71fb29095e | 236375 | 11:34:18 |
| data/k4v3_corpus/87e4e4e33a86c633729fd6e1fdff0513-1.json.gz | c917293b92938f984036ba9b5575f1333f1f10d9 | 257283 | 11:34:20 |
| data/k4v3_corpus/8ab5d2164a1094e59fd1ae6fc61640e4-1.json.gz | a3d657111ade2b3fd5f72da6b1495677875aca7f | 244949 | 11:34:22 |
| data/k4v3_corpus/8cffa18cf7a23d39de702bc5ebe0b4d5-1.json.gz | 93ac1e5a1c43e3fa1f17fd59f0718b1bb5935254 | 250750 | 11:34:25 |
| data/k4v3_corpus/900129bcc8ba84a297a99a3f059c031a-1.json.gz | 8e830cf22de447debe00705920989290d75dc1c5 | 228684 | 11:34:27 |
| data/k4v3_corpus/93eb5359cd5901c6d8859997ff2b39ad-1.json.gz | b2ead75ded05a02cb422b20451bb0c56822bb0dd | 249520 | 11:34:29 |
| data/k4v3_corpus/93f475b17ec71fd4d209b2ee0e6b3bd7-1.json.gz | 4be349195e65c5fa2166950d93b667519f9ab9f7 | 241814 | 11:34:31 |
| data/k4v3_corpus/9606d0f0efd276a068339214caf9b2c7-1.json.gz | b12f227e1b2c6ef64838d5870c07b60117b98799 | 248676 | 11:34:33 |
| data/k4v3_corpus/9ddc183e2982ce2b0bd9b39d6c9eeb74-1.json.gz | 050dfbc740268a76ae2a2673ca04f984c32721d0 | 251259 | 11:34:35 |
| data/k4v3_corpus/a6a86b994c3e854ff2f72e6b4c670281-1.json.gz | 3f12d9b26266f6cae12e011457dbf1b25fde4876 | 230369 | 11:34:37 |
| data/k4v3_corpus/ac3124b39d5a03852a4a2243198c7106-1.json.gz | 11ccc985ff128075511d25466328a397e11234a6 | 229457 | 11:34:39 |
| data/k4v3_corpus/ac7a9bdb1f648cd7aced960a00bc9bbe-1.json.gz | 8b2d192ce0421d8de028d46a7b8e995f0d004441 | 244022 | 11:34:41 |
| data/k4v3_corpus/b51ab7e4d2229b1a563ede1e846c9d34-1.json.gz | 0ebba1285ca161b91428e80026cb8dc1ac23d80e | 234678 | 11:34:43 |
| data/k4v3_corpus/bac56cb08d38c48ae511b8e4e35159de-1.json.gz | 2d506806ff1002932f48ff040fd285137b52e7ca | 254952 | 11:34:45 |
| data/k4v3_corpus/c0c59c811463a2e93486b50273abc325-1.json.gz | 231b37e59048f4d9dec7a9686a6cc48ff4861dee | 234560 | 11:34:47 |
| data/k4v3_corpus/c5dae318b1cedcb30e4b5e6086b16690-1.json.gz | 6d137c351d7474a2820c2566d813093b34554479 | 248800 | 11:34:50 |
| data/k4v3_corpus/c86e22000bc828dc2809a9381c04fc32-1.json.gz | e4937d9e9f5a639fb6426528e990d955c0ee4ef9 | 247860 | 11:34:51 |
| data/k4v3_corpus/c94ca614345d8efcb793bebc202e18e1-1.json.gz | dacc7ad36a12f016183ec96b4e64a32ecc0f7774 | 251556 | 11:34:54 |
| data/k4v3_corpus/c9594d329709cedde05341d58ea88e32-1.json.gz | bbc8b23e0c4155266e8203f28f5faaa9d5c63e4a | 238379 | 11:34:56 |
| data/k4v3_corpus/c9c89163f06844a9da93c9e11b10af44-1.json.gz | be083ccb927fe97aca2aa676021bacafac77020d | 262726 | 11:34:58 |
| data/k4v3_corpus/ca1e468f93e4157803c03679c5cf5ea4-1.json.gz | a46248dc045f32d871e13e0d2a58c4a894380858 | 249483 | 11:35:00 |
| data/k4v3_corpus/cc13fb5f4f00ad3aba0f66132127315c-1.json.gz | e67af4dc689f6e667a0cf8095e8c96b7fb98e200 | 250488 | 11:35:02 |
| data/k4v3_corpus/ccb0f422beca40eaf2c8cc5c19ddf8a0-1.json.gz | 4cc191d7444f0e358fd7c3473c2024f7fc1822af | 231274 | 11:35:04 |
| data/k4v3_corpus/ccfc623be6e3608075d4ad7b77f97f87-1.json.gz | 3ca1c9bc26e96b9087a1884e1c5cd66aaaee82e2 | 252759 | 11:35:06 |
| data/k4v3_corpus/d7bfe8b3cb8a532ab81f36c02974f312-1.json.gz | 86a93b8fa46e61a85994fd83024f9d077d6384a8 | 250044 | 11:35:08 |
| data/k4v3_corpus/e1c8e1803f1844a54728f5ae91e2ffec-1.json.gz | 4dea1cfd6ffe101c66f46e9c560f7fdfbe4adadb | 247275 | 11:35:11 |
| data/k4v3_corpus/e878b12cafc3dc043aa35a9ceafbd0ad-1.json.gz | a71b3b5d930836a8842e224e8b0fceacc1620232 | 239664 | 11:35:13 |
| data/k4v3_corpus/ed2a4c8f68118ec5dddddd7186e1f3e5-1.json.gz | 6c6cc505c2d9db0e63651419cc2821cb0fc12e8d | 247831 | 11:35:15 |
| data/k4v3_corpus/ee902884b2230898b232a775c6df3435-1.json.gz | d194b7275df9f5eb1003ac498da7bb94b9b95f21 | 240220 | 11:35:17 |
| data/k4v3_corpus/efe3fcb23e2b0683c2c1481180a5514d-1.json.gz | 8df6d76ad9ad53dc0ad34b2f61d7999e0a9c2a55 | 249372 | 11:35:19 |
| data/k4v3_corpus/f5317b8b280af30508ca95563809a576-1.json.gz | 22a187e6191bb3342c804eaf4456ea0f2b7ab7f7 | 235276 | 11:35:21 |
| data/k4v3_corpus/f6a51c4949be2bd10136d0c5467a7840-1.json.gz | 3b553d8b2c72fff181fc8e1b278a04a885b104aa | 247312 | 11:35:23 |
| data/k4v3_corpus/f7d141cd125fdf0fde25d4fa6df3ddaf-1.json.gz | 4f633e21e56c117459f37c8a8d9daed95063f2e8 | 245702 | 11:35:25 |
| data/k4v3_corpus/fad061e3781af88f07a9c5411df51292-1.json.gz | c729d1e12bebe12896f2cf7fd06a4d01b216aa06 | 247875 | 11:35:27 |
| data/hakimi3pro_corpus/04ff4c7ab8f3e1cf68f96d4562a037d4-1.json.gz | e74d8d03fdf6704970e5cdde0ee876faff006459 | 250495 | 11:35:29 |
| data/hakimi3pro_corpus/08adc86655fe479dbb4053c34b2afa7f-1.json.gz | de41807978906b84f41ff5b70e8dd53c76d2d07d | 243805 | 11:35:31 |
| data/hakimi3pro_corpus/094ff8f47c4111d7437bd70bcf91820e-1.json.gz | 2acec425410d1ea75ec7a4c93272ffbde9722c82 | 236933 | 11:35:33 |
| data/hakimi3pro_corpus/0bf0a46d7d06aa1c6d102fbf816f44b5-1.json.gz | 8b6da1d5684268fa8673fb77b319bdc13644fea8 | 249157 | 11:35:35 |
| data/hakimi3pro_corpus/0c46c367e5e1d271359286f6e6935294-1.json.gz | 019c44c2b04cd4f5aac7b469240ecb49e5d88ac1 | 251538 | 11:36:02 |
| data/hakimi3pro_corpus/11b1a9a00bf5168882940d41b102dc76-1.json.gz | 3bab49f7ce1784969b9a39e6ec6e136e9b30391e | 239201 | 11:36:04 |
| data/hakimi3pro_corpus/1ac209242595196ce27630becb38cc0e-1.json.gz | 2e75f47c5fc9e3ae1ad310184001d46b07220ceb | 247473 | 11:36:12 |
| data/hakimi3pro_corpus/206f3a679a7fc558975905f92df2ee5f-1.json.gz | ff11f5fe3eec109359a64cb5c2864c644732526c | 244311 | 11:36:14 |
| data/hakimi3pro_corpus/2249eeb6e3c35d2c3255986165460d00-1.json.gz | b4ea52346caa8bc452f4464eaaac44d0cc64fa45 | 242094 | 11:36:16 |
| data/hakimi3pro_corpus/27013561f3ffd40a3750892fd1e28b57-1.json.gz | 1dc3804e5201690ec18f532d74f06bc962d88479 | 250871 | 11:36:18 |
| data/hakimi3pro_corpus/29a630f196095373f36ef9920f7db7cb-1.json.gz | 86318a4275994406044741026fb464617c41a4f0 | 247213 | 11:36:20 |
| data/hakimi3pro_corpus/2f2d7b1fc1881df214c6eec240f66963-1.json.gz | 64f03f345d6c94267c5da5a602066bd6a0af81a6 | 243351 | 11:36:22 |
| data/hakimi3pro_corpus/30a08fb2db215e9e208e3fe28d130181-1.json.gz | e6b74c8abb8af7d3b95cdb7eeda037c8220c9865 | 247371 | 11:36:25 |
| data/hakimi3pro_corpus/330583246749020dd53146f97a30a614-1.json.gz | 3f363ed0a6b9ac90f77293a4c94786396a0f959b | 248411 | 11:36:27 |
| data/hakimi3pro_corpus/3359a5d51de726b90f02b7e56a08610d-1.json.gz | dd5c730c8d948fb57f193d06c833eb6e368b8752 | 255569 | 11:36:29 |
| data/hakimi3pro_corpus/34b96600fd0cfe97e80ae3638cf1cd2f-1.json.gz | 45d2545cf5662aa5994a301ce9f299424e116736 | 247851 | 11:36:31 |
| data/hakimi3pro_corpus/3605368a0def08b2ddf3a3e6a27752a9-1.json.gz | 9ce762fd7161b60b5da71fdaa7a4afdefdb9486c | 249610 | 11:36:33 |
| data/hakimi3pro_corpus/3629c02dd407f0daedcb0f5d9f9a124a-1.json.gz | c9d605d6f35290198d909d0d5bb0f702e840eb0d | 242790 | 11:36:35 |
| data/hakimi3pro_corpus/3692156ef8d40a0c8f49731dee0db483-1.json.gz | 2df8f2fd20ad492fcd3f09087277a5c8590dde50 | 245242 | 11:36:37 |
| data/hakimi3pro_corpus/36c8ddfc108e6dad08f218886f36caa4-1.json.gz | 80496aa4376d7f6ed715b854f46b111145f5d085 | 249241 | 11:36:38 |
| data/hakimi3pro_corpus/3b24eca52f59a2b9bafe6d0d9db4b970-1.json.gz | e6589cc9d3bc178f03a338cfacbc6dd41d54662c | 251868 | 11:36:40 |
| data/hakimi3pro_corpus/3b2d12994a75c746fc26eeaa4c04b5b2-1.json.gz | 866dfacae0f6634d13445b49c4ee7a531ebb4c92 | 252909 | 11:36:43 |
| data/hakimi3pro_corpus/3f46af18dc61b3a03ab6c6d07ac3e9d4-1.json.gz | 4454648613623b6dae86a76b9ed2abb00e9ba7b4 | 254622 | 11:36:45 |
| data/hakimi3pro_corpus/42431515b2dfed5a2c10f262e65165c6-1.json.gz | 142ec65a61b1f68a045254c15e7ef5e11f385401 | 240692 | 11:36:47 |
| data/hakimi3pro_corpus/44ad80f76a56a697881982137233a0cb-1.json.gz | 06c47196628fe29b21783f9db03381e45312a534 | 245780 | 11:36:49 |
| data/hakimi3pro_corpus/44c8241135643d6fd635dfb57658404c-1.json.gz | b3590996da484432b5f4e68772b7125c33b9adb1 | 247817 | 11:36:51 |
| data/hakimi3pro_corpus/4872ae8392ea5e5478d1ff94ae3244a7-1.json.gz | 4ebe14d3dcb051c352393ed3a5933c90c56477db | 244615 | 11:36:53 |
| data/hakimi3pro_corpus/4ca64ec13b80d0310f52b93265106cbf-1.json.gz | 5737df72ebfbc3a4bd008bff94994ed5bf9966ac | 249202 | 11:36:54 |
| data/hakimi3pro_corpus/4e3f4788f02558e951c6aeb1b7c3698f-1.json.gz | a869cc2b998a9d29909bfb49e63e088e5ff1fd10 | 246433 | 11:36:57 |
| data/hakimi3pro_corpus/517a1c5024b5f567191276cecdb26982-1.json.gz | a7eebbb4e14e8cc56f3a9ac9324e778d31eb8d89 | 239743 | 11:36:59 |
| data/hakimi3pro_corpus/55b40ed07144ae9f6acb98e5480c60c9-1.json.gz | 0758f4b8502ae89d018b178e54eef7b4745060e2 | 241512 | 11:37:01 |
| data/hakimi3pro_corpus/5af302998eec4acaaa734d429a896d18-1.json.gz | 89cf2cb18d037d3e79846e8dd5c601df86a575f3 | 241832 | 11:37:03 |
| data/hakimi3pro_corpus/5d076412484c838cfcf873984412da5d-1.json.gz | d78fc8b31d4dc3b24b429342514a09b3c868d791 | 240862 | 11:37:05 |
| data/hakimi3pro_corpus/5ffeecc6a420d7b132f4e4f62c17d76a-1.json.gz | 169894629e07c94c78bbd32ceba62c2a047fd8f8 | 243376 | 11:37:07 |
| data/hakimi3pro_corpus/6515b649a33098598d0a50bf19cfc7cc-1.json.gz | 90705cae6a708cf868f01879112115e872b43cc8 | 251783 | 11:37:09 |
| data/hakimi3pro_corpus/6e8bba02b162605d41eb94a9700e25c4-1.json.gz | 7a91a056e93241485a932100d19a96f6c116bc74 | 248977 | 11:37:11 |
| data/hakimi3pro_corpus/6f5f386fba5d72bcba1481cd0d9b8c18-1.json.gz | 774004a6fd9695e293c91498a395014db2016a12 | 248502 | 11:37:13 |
| data/hakimi3pro_corpus/6fb8a15421e3e1d7e35b7067c0624c42-1.json.gz | f418e42fb83851524484ab8eb4aaa429e162b733 | 257641 | 11:37:15 |
| data/hakimi3pro_corpus/70844ff690c56772efbebd7b3315f9ec-1.json.gz | e1fbc9fe6d7b18371ef0a0900845e928e529cd25 | 238093 | 11:37:17 |
| data/hakimi3pro_corpus/7a645248893bb256df57d5139fc3796e-1.json.gz | 59aca1270e3bdaae1ffbc89a6e378844b544bdac | 254420 | 11:37:19 |
| data/hakimi3pro_corpus/7fc12203e8a4ccea455a111d1610b4a3-1.json.gz | 7303db47386d75c1fb1aa6c7e83ef29cf2d20965 | 258248 | 11:37:21 |
| data/hakimi3pro_corpus/82f5ab7bbe9b3a0c7871614f34e38a9e-1.json.gz | 8ee8bef38264ef2bb3ed75d6ec4243e76bc5caf8 | 244606 | 11:37:26 |
| data/hakimi3pro_corpus/82faf124f687f3e88a4c64e8d134cf68-1.json.gz | 5df47c91e0ef8ee5811c18d143a1378ca13b1137 | 236404 | 11:37:27 |
| data/hakimi3pro_corpus/8cffa18cf7a23d39de702bc5ebe0b4d5-1.json.gz | 93ac1e5a1c43e3fa1f17fd59f0718b1bb5935254 | 250750 | 11:37:30 |
| data/hakimi3pro_corpus/92f4121b34a5ca8eb205a9ba519f7e27-1.json.gz | 0a83e2394cbd1b73a94520cc40a2801ef8a1a970 | 251694 | 11:37:32 |
| data/hakimi3pro_corpus/92fa9e469ebc8764fa583dcb24ca8123-1.json.gz | 057619d864c5ac5dd3105d7f49f88f8c461cbd00 | 252952 | 11:37:34 |
| data/hakimi3pro_corpus/95c7259f54543378ac0771795956def0-1.json.gz | 26d0234b018161db3a68b83504e4a484168f3764 | 255293 | 11:37:36 |
| data/hakimi3pro_corpus/99521eaec4c5f4d1d7413fd540a463f0-1.json.gz | 943bfc581205feec29034a65fc1f7585859bdbfe | 244445 | 11:37:38 |
| data/hakimi3pro_corpus/a5b3312463abf0b6a7d597b1de355e78-1.json.gz | b68da87a31fff1d23b2c1095dc0ce9c0df35a77e | 251361 | 11:37:40 |
| data/hakimi3pro_corpus/ad3765562c64f74dfac72e5c024fcd6c-1.json.gz | e502c91f304cc7d78cb5ba8be1536ccd10dbdc23 | 256360 | 11:37:42 |
| data/hakimi3pro_corpus/b47b01ee39e0d7834d4e4d5209d8253b-1.json.gz | 7b59246a8e826294d3ade57c698be3f8311d82b1 | 246061 | 11:37:44 |
| data/hakimi3pro_corpus/b5126d7ccec08e8354bd0802044a94cc-1.json.gz | adbc9901f0c289eafc4b5b6517494ff65d889cf9 | 249746 | 11:37:46 |
| data/hakimi3pro_corpus/bd41fe583241ecffdfd5ad628ea63a5b-1.json.gz | 8bbe9bf010717c7c3de692b9c05e72bc0ca68fdb | 238996 | 11:37:48 |
| data/hakimi3pro_corpus/bd90dd09f26895c70d438521df8b8f63-1.json.gz | dbed13e71877b933e0070ff73f8b9a4daa0bac5a | 246850 | 11:37:50 |
| data/hakimi3pro_corpus/cda06eca8d1e0ad2febd2332823ee044-1.json.gz | 8736f92e18e9b127213b6f31cf4bc2f9d60093f2 | 242474 | 11:37:53 |
| data/hakimi3pro_corpus/d2299276dacb4c88005c80f00ab31e40-1.json.gz | 74966c59d5df49e1541c9ff698bf65c80226da08 | 252996 | 11:37:55 |
| data/hakimi3pro_corpus/d27d760e2949cdf734f84849c2f6e919-1.json.gz | 9e0f525d884a8829a66fa7910678f69dfd89bdc4 | 242849 | 11:37:57 |
| data/hakimi3pro_corpus/d57ab5d0f223e0dc36d357a618535c0f-1.json.gz | ed91ec003d4e7565a87fe9cdb5831ea9238763a3 | 237549 | 11:37:59 |
| data/hakimi3pro_corpus/df6c25f3864459415b1c22672f839cbf-1.json.gz | f00dabac19ac41039ab44c48ff0a232aebf9ed3f | 236292 | 11:38:01 |
| data/hakimi3pro_corpus/e03c99f934609b2055de81f7b6b59415-1.json.gz | 664a7e51d105b1676723fbf2f1ac3026f7ea160e | 241848 | 11:38:03 |
| data/hakimi3pro_corpus/e326362d8accc2b22ed6d77a7ca09116-1.json.gz | fbccbe7e3a9d27dd4f400032359e1979c59172e9 | 241568 | 11:38:06 |
| data/hakimi3pro_corpus/ed2a4c8f68118ec5dddddd7186e1f3e5-1.json.gz | 6c6cc505c2d9db0e63651419cc2821cb0fc12e8d | 247831 | 11:38:08 |
| data/hakimi3pro_corpus/f50e61a5642af6659c9677f47ff51a60-1.json.gz | 79e8c68a66eced2c78fc54e5d7de64c14177793a | 232026 | 11:38:10 |
| data/hakimi3pro_corpus/fb3e743b714d771c683d2919c0e8e48e-1.json.gz | aca0eeda3537f7ecd169372673bb8081682c8464 | 229642 | 11:38:12 |
| data/fable6_corpus/028149c8e58f52d631adada01ea40248-1.json.gz | 043c9aec22eaf89801645c5b33be2cb34bcfe835 | 258845 | 11:38:14 |
| data/fable6_corpus/086b08c2836a2b76e19bb67ea0614375-1.json.gz | 4e5e359c9e178d4ad2d98950e9624bd3f5b63486 | 247807 | 11:38:16 |
| data/fable6_corpus/0bf0a46d7d06aa1c6d102fbf816f44b5-1.json.gz | 8b6da1d5684268fa8673fb77b319bdc13644fea8 | 249157 | 11:38:18 |
| data/fable6_corpus/0db9b8b9ed673da514aab930f57c1358-1.json.gz | d86b3da7ad835678f9877ef4365151c462742285 | 241244 | 11:38:20 |
| data/fable6_corpus/18077e6fec29f70b6b0141da7b6ee3c9-1.json.gz | 963af296cac889067d657d8021df2d248003b171 | 226998 | 11:38:22 |
| data/fable6_corpus/1a1051dee9b88689fb426f2ea4cbd96f-1.json.gz | 963cc1becded0c25fb2f8ee2e71c911bb343f4a0 | 239101 | 11:38:24 |
| data/fable6_corpus/1bf2c78c807ad42fe767eb37ad7b61ca-1.json.gz | 0ce2d442271be6b52d56c2de3a2050e84efcf20d | 245611 | 11:38:26 |
| data/fable6_corpus/1c2090317b8de8624176fbfbfb801345-1.json.gz | 1a5f69a8cec7dbb0ad9472712206475ce7447dc0 | 240975 | 11:38:28 |
| data/fable6_corpus/1d2a983bf882f255b2a01487fe7034d9-1.json.gz | 34c81fea58b8c984224cec3befc00d33ac2c65db | 239993 | 11:38:30 |
| data/fable6_corpus/1da23f57b5c4dac4ff726a8b65fa3e4a-1.json.gz | 4004ab3dd81710cff57935456ac16b2c798efdd1 | 239748 | 11:38:32 |
| data/fable6_corpus/1da3659f35017f4956133a839faf4e19-1.json.gz | f789e1f6f0ae69ca4d1d23b3ba7a1b2831c5314a | 230433 | 11:38:34 |
| data/fable6_corpus/2f195cf12192a8dffd0e7cfab569c7e9-1.json.gz | 40dd4c1ddfda030ccb1d954efd339de9f1de9fce | 232131 | 11:38:36 |
| data/fable6_corpus/30ddd7a614ee06f0980c046a94b2bc0a-1.json.gz | 95937da285482fa0ea0889ab0e70e12e5b6d6528 | 239155 | 11:38:40 |
| data/fable6_corpus/319bf8017e02a2eb1bf3385ca9fff01b-1.json.gz | 921ce4d5ea0ccbbd16b1b49b66a98d8a5c3c8baa | 247489 | 11:38:42 |
| data/fable6_corpus/3605368a0def08b2ddf3a3e6a27752a9-1.json.gz | 9ce762fd7161b60b5da71fdaa7a4afdefdb9486c | 249610 | 11:38:44 |
| data/fable6_corpus/3b4352764cd9b25b84fc7b7e1b856dfb-1.json.gz | 965d0dc66e64d6d5190e0da6cf998a5610342648 | 254003 | 11:38:46 |
| data/fable6_corpus/3c33ccc4e35444b7286247a574d07f7c-1.json.gz | e28095895acd7974b3a35a87a6960c6042c21532 | 239625 | 11:38:48 |
| data/fable6_corpus/41667b44f2cd28923a8f2849399e2f20-1.json.gz | 3d1c01a0b66f7ee6968e54f6334dfb1953f4ba4f | 238152 | 11:38:51 |
| data/fable6_corpus/419dafd46e0c2a6ecec1fb17a121e41e-1.json.gz | 95d6ce35f480d71eaa5ddb2fd877acbacaa0c636 | 248791 | 11:38:53 |
| data/fable6_corpus/4622413f5811cd678b101bf7a70392da-1.json.gz | 6a7ea41bb426ed66b48667f0654fc607ab347511 | 236300 | 11:38:55 |
| data/fable6_corpus/4666e2fe48d6735b9b37d170626eb2e6-1.json.gz | 58e210285e3d8d93dbffbfec61ea17f025c05c3d | 245936 | 11:38:57 |
| data/fable6_corpus/4a717c678bf551229f531e5983275e05-1.json.gz | 8637c4e1eacf8bbe78206b53a0819ff8a87902ee | 244188 | 11:38:59 |
| data/fable6_corpus/4bba30fe415748a99bc5b29b039c8146-1.json.gz | c0b8d6103383113e3b4fb47574d5504e2ee7ffc4 | 234072 | 11:39:01 |
| data/fable6_corpus/4e8fc9a4eae9a81f19f867e8002f557f-1.json.gz | afdf1157be47f82df8c8ff0f2a31f3395f8fd7c2 | 236610 | 11:39:03 |
| data/fable6_corpus/532502cb48273690f0134fdc1acc939a-1.json.gz | 13990550788486ef5a52260d774f2ea5d8ce6f40 | 250511 | 11:39:05 |
| data/fable6_corpus/5390778c3c106770b09132c171f8fc2e-1.json.gz | 35ccb95036343e76240dd4807afefb11acb25cd1 | 250661 | 11:39:07 |
| data/fable6_corpus/57dd17fcd5789c846ccd91ec540dbc52-1.json.gz | 9692f89e4d4d9d398e37ef4bd07fc45b26719d16 | 238886 | 11:39:10 |
| data/fable6_corpus/5a196aa996d838f3eefedebfcf236ddc-1.json.gz | bd4aa81c9f8f2b5aa432a594aa7ae9fbc00a1ec6 | 241143 | 11:39:12 |
| data/fable6_corpus/5ab766b9b4a00e90dd1def6acf4f7f65-1.json.gz | d8826beebea0317c948b708acb1dccf4152fdeec | 245915 | 11:39:14 |
| data/fable6_corpus/61c4be599f23381d5940a535b7d7a7bb-1.json.gz | 93814d9727e03661d923a9d4090df20fc27ac0b4 | 238817 | 11:39:15 |
| data/fable6_corpus/62ea688e953b67a009e1271395cd0ea4-1.json.gz | e743429dac75618198bcf5c90b77c3f7fb47b0f9 | 231001 | 11:39:17 |
| data/fable6_corpus/677f8530c67913e4ef0017cea9f690bf-1.json.gz | b8faf6fedb62a2f9ba5dceda934f7f66c37c0c57 | 251355 | 11:39:20 |
| data/fable6_corpus/68c1fcc6776d677593190d0e0ef09465-1.json.gz | eacbcc006b7b21a1dc47789a195a965a6ee96807 | 239689 | 11:39:22 |
| data/fable6_corpus/6a0f495baada46d66d0dcd3aa8252c97-1.json.gz | fdbde9fd56a64beb1e8a1e85d0dd3d74d2e883ee | 241557 | 11:39:24 |
| data/fable6_corpus/6bf0b5fed3388d818e9687d81eed4a0a-1.json.gz | 73a1becad259c5673236581808efbff9470a48c9 | 245799 | 11:39:25 |
| data/fable6_corpus/6e99c680697850be83ea92ec3e993651-1.json.gz | 9466d9b0018f2993d172b0cc90b56ec0dd529209 | 241559 | 11:39:27 |
| data/fable6_corpus/6ef0bb980d2c3ead86b35cfbee87f816-1.json.gz | 5bbf1da9542f44629668210743aec5ba30308eff | 238956 | 11:39:30 |
| data/fable6_corpus/72f3c476adf0cda5cab30591d2b6d51e-1.json.gz | c86cb18fba76acd454764e6639fa1a9dd3b22d35 | 238095 | 11:39:32 |
| data/fable6_corpus/742dd2d1c9fde58059620f81bf267ddb-1.json.gz | 4c88ccedce63ef24dc9392dee86822a4e030de6a | 246710 | 11:39:34 |
| data/fable6_corpus/7d4509df9f9ed8bdc3f4e13691f8b011-1.json.gz | 5fe114fa55b44f33f04858dfca0a502dfff5e928 | 248674 | 11:39:36 |
| data/fable6_corpus/88bb4b4ccb3062681170b99a7139dc17-1.json.gz | 605205838cd04ab0f168eb41069c8944c6919c85 | 251373 | 11:39:38 |
| data/fable6_corpus/8c088cd9e9382517dedfb7fd9cd75bea-1.json.gz | 2e7533a90758ba9e3c06ed399f4aa2624269574a | 233029 | 11:39:40 |
| data/fable6_corpus/8f3ac445307a1922daa9649482b1c4c4-1.json.gz | 842898f557582b379a7b15f597a441e0c13fdd85 | 233971 | 11:39:42 |
| data/fable6_corpus/8fc4d7dddb5b3aac01b497ff0fbcc1e7-1.json.gz | a6f49e765550936fd90e7da531b9fd02f5be9e65 | 236164 | 11:39:44 |
| data/fable6_corpus/93f80e34d20782ff94b8b9f07271fc7f-1.json.gz | bdad1eb0f271fd4e1119bb9aee68ded27fa08c5b | 246618 | 11:39:46 |
| data/fable6_corpus/9997cb4f130eaa0cc29c10e07845e2f4-1.json.gz | e838471f12396115820a258973581ead97068be6 | 243006 | 11:39:48 |
| data/fable6_corpus/9c7d218996ac3ef353fb1592358bd819-1.json.gz | 56b624dbf1a492d9e27a21104607d4400c40ac67 | 252571 | 11:39:50 |
| data/fable6_corpus/9ddc183e2982ce2b0bd9b39d6c9eeb74-1.json.gz | 050dfbc740268a76ae2a2673ca04f984c32721d0 | 251259 | 11:39:55 |
| data/fable6_corpus/a9ce243fddbd3f1ad3dca351d38d5dcb-1.json.gz | 17914138a129c523cea4049a6fe8a9249e6972ef | 252520 | 11:39:57 |
| data/fable6_corpus/adaeaf283afe7809297ea12d6cf4eb70-1.json.gz | bb253716f7ede9785480a3f6d3292812e13cb308 | 243974 | 11:40:00 |
| data/fable6_corpus/afd73399acc0549c82b97cc575f6b7a6-1.json.gz | ac287059d292f9ebd94b9962ff8c076e34373107 | 227592 | 11:40:01 |
| data/fable6_corpus/b33681c4fbdfb8d0406554432ff7bfef-1.json.gz | 059bf9ac51125b87f8c1c0314f3e2ef6d9f82df0 | 237181 | 11:40:03 |
| data/fable6_corpus/b47b01ee39e0d7834d4e4d5209d8253b-1.json.gz | 7b59246a8e826294d3ade57c698be3f8311d82b1 | 246061 | 11:40:06 |
| data/fable6_corpus/b67e5ccc51bb7e34c4b73904133bd53f-1.json.gz | 9520bd6a552c8983e91aaf5c2a49033e4bfe8158 | 249030 | 11:40:08 |
| data/fable6_corpus/be79ff4962ed78ebdfdf72abf0ea9bba-1.json.gz | 72450053aec7e7b341f7131cdef0d9e7ebc2691e | 239807 | 11:40:10 |
| data/fable6_corpus/c567cb7c9ad453eadcbe34545a28dc7b-1.json.gz | ca7d05ce29c67fa22c20efdecc66af80ae11318f | 236790 | 11:40:12 |
| data/fable6_corpus/c5b82ad3658c54a9385a801e6c8ebae8-1.json.gz | 8012f3704c24673a97be8261d5c4d618efb2f16e | 239046 | 11:40:14 |
| data/fable6_corpus/c9033c88211282ec4d62e87f67d6ef73-1.json.gz | b4fbe3a5f9defccac51dfa7ed3ca83f2bd15c23a | 248223 | 11:40:16 |
| data/fable6_corpus/cc13fb5f4f00ad3aba0f66132127315c-1.json.gz | e67af4dc689f6e667a0cf8095e8c96b7fb98e200 | 250488 | 11:40:18 |
| data/fable6_corpus/cfbf9f82d35d1d5bca3f23bf336ad88f-1.json.gz | 89f629367ecebb9b5ac5e2799efcbf68ec807256 | 243423 | 11:40:20 |
| data/fable6_corpus/d57ab5d0f223e0dc36d357a618535c0f-1.json.gz | ed91ec003d4e7565a87fe9cdb5831ea9238763a3 | 237549 | 11:40:22 |
| data/fable6_corpus/d93131003397b3b660d8eb2940ee66cf-1.json.gz | 275acdab1912266c4626e6df2666a1d3a44b9a8c | 248496 | 11:40:24 |
| data/fable6_corpus/d93ce9c173afa12f441645ad329b2d0b-1.json.gz | c389abad36cf334b2bd5337cd9f09c8fb07d03a5 | 243507 | 11:40:26 |
| data/fable6_corpus/d99b9c6b208cfda6be462863124adbd7-1.json.gz | 2b596654260beea61f07d91355550f8f2385a0ca | 235887 | 11:40:28 |
| data/fable6_corpus/db661dc094f834429239a2c93bc35891-1.json.gz | 6b107597629d7efd0c440e9d6a508fbf416d627b | 233791 | 11:40:30 |
| data/fable6_corpus/db687a767fadd784c2770fd59bc3b4ea-1.json.gz | 667eb71967c2389b1c8808fe7323977ace38025f | 230880 | 11:40:33 |
| data/fable6_corpus/db74c2b889b34a120c96811dd5f1e54f-1.json.gz | 59167c155bfd406d592d87a92612e715b9c84798 | 242655 | 11:40:35 |
| data/fable6_corpus/e227b3a48b2c0cd296f9576392dcb6d7-1.json.gz | 487f3a1505eea9e9eb23a50e9f800abd9f25ee22 | 249104 | 11:40:37 |
| data/fable6_corpus/e326362d8accc2b22ed6d77a7ca09116-1.json.gz | fbccbe7e3a9d27dd4f400032359e1979c59172e9 | 241568 | 11:40:39 |
| data/fable6_corpus/e3caf092d8bdaabd57215a5ad02e945b-1.json.gz | a8b7ae3068c1c6d1770ef272a6e6399fb3716daa | 234002 | 11:40:41 |
| data/fable6_corpus/e9e0d3229455f9ef63da99f301379a1b-1.json.gz | a7b3a8c4c2c9969d1fabf2faf00857ed128b08c9 | 227163 | 11:40:42 |
| data/fable6_corpus/ea76e968e3aa562b846056be51f36307-1.json.gz | 431be14d3607d2648917d84b24013f39061ec1ed | 255650 | 11:40:44 |
| data/fable6_corpus/f4d9f50bcdd28b1c48cfc397d91857ad-1.json.gz | 40cebc5e73801b023496c186c3cedd41774bf309 | 250929 | 11:40:47 |
| data/fable6_corpus/f59c82f15aa5aa96961762e1b415e606-1.json.gz | 85fe3ec6764f56319bfd4ebf253e32604261c4dd | 232360 | 11:40:48 |
| data/fable6_corpus/fbe863c90cfdede9b06ae78fc48de82d-1.json.gz | 10e12984db9f1aad61529cd29592560cd4a4e23b | 232238 | 11:40:50 |

## read-back verification (2026-11-13 ~19:42, gh_api GET on main)
- data/k4v3_corpus: 64 entries, all .json.gz; md5 spot-check 3/3 match staged copies
- data/hakimi3pro_corpus: 64 entries, all .json.gz; md5 spot-check 3/3 match
- data/fable6_corpus: 75 entries, all .json.gz; md5 spot-check 3/3 match
- matches.json in repo: k4v3 65 games vs 64 replays; hakimi3pro 64 vs 64; fable6 76 vs 75
  (k4v3/fable6 manifests are 1 game ahead — the 2 replays pending DGX top-up, see matches.json decision above)
- Uploads: 203/203 ok, 0 failed, 0 skipped, no 403 rate-limit hit. ~215 API calls total.
