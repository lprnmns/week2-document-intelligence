# ADR-003: Ingestion version activation ve retention

## Bağlam

Yeni PDF sürümü indekslenirken yarım veya bozuk point'lerin sorgulara görünmemesi gerekir. Aynı dosya tekrar yüklendiğinde duplicate üretmek de istenmez.

## Alternatifler

1. Point'leri doğrudan active collection'a yazmak.
2. Yeni version'ı stage edip doğrulama sonrası active pointer'ı değiştirmek.

## Karar

Content hash + pipeline fingerprint document/version identity için kullanılacak. Yeni version önce staged/inactive yazılacak; point count, schema ve metadata doğrulamasından sonra active yapılacak. Başarısız iş eski active version'ı değiştirmeyecek.

## Ölçüm/kanıt

Üç tekrar upload, retry, process restart ve başarısız parse testleri; point count ve active version snapshot'ı ile kanıtlanacak.

## Bilinen sınır

Retention süresi ve eski version silme politikası tenant hacmi ve yasal saklama ihtiyacına göre netleştirilecek.

## Section marker profili

Bilinen doküman aileleri için section marker profili pipeline fingerprint'in
parçasıdır. `none` genel PDF'lerde yapılandırılmış, karakter sınırı olan parent
pencereleri kullanır; Week 1 mentor ailesi `mentor_program_v1`, Alperen'in 28 sayfalık Week 2
program PDF'i `mentor_program_week2_v1` seçildiğinde aynı byte içeriği farklı
section-aware version olarak stage → verify → activate edilir. Profil bilinmeyen
bir değerse configuration aşamasında açıkça hata verir.
