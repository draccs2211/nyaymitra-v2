// ISO 3166-2:IN codes sent as QueryRequest.jurisdiction (backend accepts ^IN(-[A-Z]{2})?$).
// "IN" = central law only. State-specific corpus content currently exists for IN-UP only.
export const JURISDICTIONS: Array<{ code: string; en: string; hi: string }> = [
  { code: "IN", en: "Central law only", hi: "केवल केंद्रीय कानून" },
  { code: "IN-UP", en: "Uttar Pradesh", hi: "उत्तर प्रदेश" },
  { code: "IN-DL", en: "Delhi", hi: "दिल्ली" },
  { code: "IN-MH", en: "Maharashtra", hi: "महाराष्ट्र" },
  { code: "IN-KA", en: "Karnataka", hi: "कर्नाटक" },
  { code: "IN-TN", en: "Tamil Nadu", hi: "तमिलनाडु" },
  { code: "IN-WB", en: "West Bengal", hi: "पश्चिम बंगाल" },
  { code: "IN-GJ", en: "Gujarat", hi: "गुजरात" },
  { code: "IN-RJ", en: "Rajasthan", hi: "राजस्थान" },
  { code: "IN-MP", en: "Madhya Pradesh", hi: "मध्य प्रदेश" },
  { code: "IN-BR", en: "Bihar", hi: "बिहार" },
  { code: "IN-HR", en: "Haryana", hi: "हरियाणा" },
  { code: "IN-PB", en: "Punjab", hi: "पंजाब" },
  { code: "IN-KL", en: "Kerala", hi: "केरल" },
];
