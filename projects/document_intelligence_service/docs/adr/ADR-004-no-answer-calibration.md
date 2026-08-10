# ADR-004: No-answer sinyalleri ve threshold kalibrasyonu

## Bağlam

Tek bir cosine threshold, farklı query türleri ve dillerde güvenilir değildir. Benzer görünen fakat soruyu desteklemeyen evidence yanlış cevap üretebilir.

## Alternatifler

1. Tek dense cosine eşiği.
2. Multi-signal answerability gate.

## Karar

Gate; evidence boşluğu, calibrated final/rerank score, top-1/top-2 margin, evidence coverage ve ACL/document filter sonucunu birlikte kullanacak. Threshold'lar golden validation split üzerinde kalibre edilecek; test split karar vermek için kullanılmayacak. Direct prompt injection ise score threshold'ı değil, retrieval'dan önce çalışan ayrı `PromptSafetyPolicy` kararıdır.

İlk dikey dilimde `AnswerabilityPolicy` bu sinyalleri framework bağımsız olarak üretiyor. Hybrid RRF adayları RRF sırasındayken dense top-score/margin hesabının yanlış sırayı kullandığı tespit edildi; karşılaştırılabilir score kind içinde sıralama düzeltildi. Direct injection için policy retrieval'dan önce çalışıyor ve `SECURITY_POLICY` dönen vakalarda score üretmiyor. Final corpus snapshot'ındaki score-bearing, security dışı 9 validation vakasında false-negative maliyeti `3.0` ile seçilen dense score threshold `0.337857395` (`0.338`) oldu. Bu küçük validation alt kümesi güçlü genelleme kanıtı değildir ve corpus/model/chunk değişince yeniden kalibrasyon gerekir.

Sparse `0.1`, rerank `-5.0`, margin `0.0` ve coverage `0.0` henüz aynı yöntemle kalibre edilmedi; bunlar provisional değerlerdir. Eşikler `DIS_ANSWERABILITY_*` ayarlarıyla değiştirilebilir. Test split yalnız final rapor içindir ve threshold seçimine giremez.

Validation threshold'ı injection riskini tek başına çözmedi; bu nedenle direct injection artık ayrı `PromptSafetyPolicy` ile retrieval'dan önce kesiliyor. Frozen test'te prompt-injection `2/2` geçti, fakat bu modelin veya rule setinin genel olarak güvenli olduğu anlamına gelmez. No-answer gate, prompt safety, structured prompt, source provenance ve output validation defense-in-depth zincirinin ayrı katmanlarıdır.

Frozen final-corpus test split'inde runtime `0.338` ile 14 vaka değerlendirildi;
false positive `0`, false negative `2` olarak raporlandı. Reason dağılımı
`ANSWERED=10`, `LOW_RELEVANCE=1`, `SECURITY_POLICY=3` şeklindedir. Prompt
injection vakaları score false-negative olarak değil, security policy kararı
olarak değerlendirilir. Test sonucu threshold'u geriye dönük seçmek için
kullanılmadı.

Final stabilization sonrasında ürün policy seçimi de calibration scope'a bağlandı.
`mentor_program_v1` için yukarıdaki `0.338` policy aynen korunur. `generic_v1`
kanıt metadata'sı tek ve uniform olduğunda, aynı dense model/sparse encoder,
Hybrid RRF ve reranker-off yapılandırması üzerinde ayrı bir küçük validation
setinden seçilen dense `0.247` ve coverage `0.367` değerleri kullanılır.
Kanıt profil metadata'sı yoksa veya sonuçlarda profiller karışıyorsa sistem
generic eşiği tahmin etmez; konservatif mentor/default policy'ye döner.
Bu, bir generic PDF'nin mentor threshold'u yüzünden yanlışlıkla reddedilmesini
önlerken frozen mentor benchmark'ının calibration'ını değiştirmez.

Generic calibration artifact'ı
`eval/results/generic_document_answerability_v1.json` altında; kaynak set
`data/evaluations/generic_document_answerability_v1.jsonl` dosyasındadır.
Threshold selection yalnız `validation` split'inde yapılır; `test` split'i
sonuç raporu içindir. Bu küçük set, model/corpus/chunking değişikliklerinde
yeniden kalibrasyon gerektiren ürün policy'si olarak değerlendirilmelidir,
mentor benchmark'ının yerine geçmez.

Near-miss düzeltmesi calibration threshold'larını değiştirmeden eklendi.
Lexical `coverage_ratio` yalnız konu benzerliğini ölçmeye devam eder; soru
`2024` isterken evidence içinde yalnız başka bir programa ait `2024` bulunması
artık qualifier coverage olarak kabul edilmez. Yıl ve yüzde qualifier'ları,
sorunun konu anchor'ı öncesindeki bounded bağlamla ilişkilendirilir; yıl
qualifier'ı ayrıca istenen yüksek-güvenli attribute (`kapanış`, `kontenjan`,
`ücret` vb.) ile aynı yerel kayda ait olmalıdır. Açık tırnak içi terimler de
exact normalized phrase olarak doğrulanır. Bu, NER/LLM classifier değildir ve
belirsiz doğal dil attribute'larını çözme iddiasında bulunmaz.

Generic dataset'in untouched test'i 6 vakaya genişletildi: bir answerable ve
beş unanswerable/near-miss vaka; wrong-year, wrong-program, wrong-discount ve
year-attribute mismatch vakaları `INSUFFICIENT_COVERAGE` olarak reddedildi.
Test sonucu `TP=1`, `TN=5`, `FP=0`, `FN=0` oldu. Dense `0.247` ve lexical
coverage `0.367` aynen korunmuştur; bu sonuçlar threshold seçiminde
kullanılmadı.

## Ölçüm/kanıt

False answer, false rejection, no-answer precision/recall ve LLM skip rate raporlanacak.

## Bilinen sınır

Calibration setindeki leakage veya dengesiz answerable/unanswerable dağılımı sonucu bozabilir. Qualifier kontrolü yalnız açık yıl/yüzde/tırnak ifadelerini ve küçük attribute sözlüğünü kapsar; genel semantik soru-anlamlandırma yerine geçmez.
