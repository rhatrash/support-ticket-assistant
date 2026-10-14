"""Generate a small synthetic bilingual (English / Arabic) support-ticket dataset.

Why synthetic? Real customer tickets are private, and the goal here is to demonstrate an
end-to-end NLP system. ~12% of the tickets are deliberately "hard" (mixed intents,
code-switching) so that evaluation and error analysis are meaningful.

Labeling rule for hard tickets: label by what the customer most needs resolved.

Run:  python -m src.make_data
"""
import random

import pandas as pd

from .config import DATA_DIR, SEED

PRODUCTS = ["Najm Plus", "Tayseer", "Safa Pay", "Rawaq Cloud"]

# (category, intent, English template, Arabic template)
TEMPLATES = [
    # ---- billing
    ("billing", "double_charge", "I was charged twice for my {product} subscription.", "تم خصم المبلغ مرتين مقابل اشتراك {product}."),
    ("billing", "unknown_charge", "There is an unknown charge of {amount} SAR on my statement.", "يوجد خصم غير معروف بقيمة {amount} ريال في كشف حسابي."),
    ("billing", "wrong_invoice", "My invoice number {order} shows the wrong amount.", "الفاتورة رقم {order} فيها مبلغ غير صحيح."),
    ("billing", "charge_after_cancel", "Why was I billed for {product} after I cancelled it?", "ليش انخصم مني مبلغ {product} بعد ما ألغيت الاشتراك؟"),
    # ---- technical
    ("technical", "app_crash", "The {product} app crashes every time I open it.", "تطبيق {product} يقفل لحاله كل ما أفتحه."),
    ("technical", "login_error", "I get error code {code} when I try to sign in.", "يطلع لي خطأ رقم {code} لما أحاول أسجل دخول."),
    ("technical", "slow_performance", "{product} is extremely slow and keeps freezing.", "{product} بطيء جدا ويعلق باستمرار."),
    ("technical", "sync_failure", "My data is not syncing between my phone and laptop.", "بياناتي ما تتزامن بين جوالي واللابتوب."),
    # ---- account
    ("account", "password_reset", "I did not receive the password reset email.", "ما وصلني إيميل إعادة تعيين كلمة المرور."),
    ("account", "hacked_account", "Someone accessed my account without my permission.", "أحد دخل على حسابي بدون إذني."),
    ("account", "change_email", "How can I change the email address on my account?", "كيف أقدر أغيّر البريد الإلكتروني في حسابي؟"),
    ("account", "delete_account", "I would like to permanently delete my account.", "أبغى أحذف حسابي نهائيا."),
    # ---- shipping
    ("shipping", "late_delivery", "My order {order} has not arrived and it is {days} days late.", "طلبي رقم {order} ما وصل وتأخر {days} أيام."),
    ("shipping", "wrong_item", "I received the wrong item in order {order}.", "وصلني منتج غلط في الطلب رقم {order}."),
    ("shipping", "damaged_item", "My package arrived damaged.", "الشحنة وصلتني تالفة."),
    ("shipping", "track_order", "Where can I track order {order}?", "وين أقدر أتتبع الطلب رقم {order}؟"),
    # ---- refund
    ("refund", "refund_status", "I returned order {order} two weeks ago and still have no refund.", "رجّعت الطلب رقم {order} من أسبوعين وما استرجعت فلوسي."),
    ("refund", "request_refund", "I want a refund for {product}, it does not work as described.", "أبغى استرجاع مبلغ {product} لأنه ما يشتغل زي الوصف."),
    ("refund", "cancel_order", "Please cancel my order {order} before it ships.", "أرجو إلغاء الطلب رقم {order} قبل الشحن."),
    ("refund", "return_policy", "What is your return policy for {product}?", "وش سياسة الإرجاع لـ {product}؟"),
]

CATEGORY_WEIGHT = {"billing": 1.0, "technical": 1.2, "account": 0.8, "shipping": 1.3, "refund": 0.6}  # mild imbalance

# (category, language, text): ambiguous / mixed-intent / code-switched tickets
HARD = [
    ("refund", "en", "I was charged for an order that never arrived and I want my money back."),
    ("technical", "en", "The app froze during payment and I was charged but got nothing."),
    ("account", "en", "My account was hacked and someone placed an order using my card."),
    ("refund", "en", "Please cancel my subscription and refund the last charge."),
    ("shipping", "en", "I paid for express delivery but the package is still not here, where did the extra fee go?"),
    ("billing", "en", "I cancelled and returned the item but I am still being billed monthly."),
    ("shipping", "ar", "انخصم مني المبلغ والطلب ما وصل لين الحين."),
    ("billing", "ar", "ما قدرت أسجل دخول وانخصم مني اشتراك جديد."),
    ("refund", "ar", "أبغى أرجّع الطلب لأن المنتج وصل مكسور."),
    ("technical", "ar", "الـ app يعلق لما أحاول أدفع payment."),
    ("account", "ar", "حسابي انسرق وأحد سوى طلبات بدون علمي."),
    ("shipping", "ar", "دفعت رسوم شحن سريع والطلب لسا ما وصل، وين راحت الرسوم؟"),
]

URGENCY = {
    "high": {
        "en": ["This is urgent.", "I need help immediately!", "Please respond ASAP, my business is stopped."],
        "ar": ["الموضوع عاجل.", "أحتاج مساعدة الآن لو سمحتم!", "ضروري تردون بأسرع وقت، شغلي متوقف."],
    },
    "medium": {
        "en": ["Please help.", "Could you look into this?", "Thanks for your help."],
        "ar": ["أرجو المساعدة.", "ممكن تشيكون على الموضوع؟", "شكرا لتعاونكم."],
    },
    "low": {
        "en": ["No rush, just asking.", "When you have time, thanks.", "Not urgent at all."],
        "ar": ["مو مستعجل، بس استفسار.", "في أي وقت يناسبكم، شكرا.", "الموضوع مو عاجل."],
    },
}

# "Historical resolutions" used to ground the drafted reply (synthetic).
RESOLUTIONS = {
    "billing": {
        "en": "We reviewed your billing history and will refund any duplicate or incorrect charge within 5-7 business days. You will receive a confirmation email once it is processed.",
        "ar": "راجعنا سجل الفوترة لديك، وسنعيد أي مبلغ مكرر أو غير صحيح خلال 5-7 أيام عمل، وسيصلك بريد تأكيد عند اكتمال العملية.",
    },
    "technical": {
        "en": "Please update the app to the latest version, restart your device and sign in again. If the issue continues, send us a screenshot and your device model so our engineers can investigate.",
        "ar": "نرجو تحديث التطبيق لآخر إصدار وإعادة تشغيل الجهاز ثم تسجيل الدخول مرة أخرى. إذا استمرت المشكلة أرسل لنا لقطة شاشة ونوع جهازك لنحيلها للفريق التقني.",
    },
    "account": {
        "en": "For your security we have verified your account. Please use the 'Forgot password' link to reset your credentials and enable two-factor authentication. Our team will contact you if we find any suspicious activity.",
        "ar": "حرصا على أمان حسابك تحققنا منه. استخدم رابط «نسيت كلمة المرور» لإعادة تعيين بياناتك وفعّل التحقق بخطوتين، وسيتواصل معك فريقنا إذا وُجد أي نشاط مشبوه.",
    },
    "shipping": {
        "en": "We are sorry for the delay. We have contacted the carrier about your order and will send you updated tracking details within 24 hours, or ship a replacement if it cannot be delivered.",
        "ar": "نعتذر عن التأخير. تواصلنا مع شركة الشحن بخصوص طلبك وسنرسل لك تفاصيل التتبع المحدّثة خلال 24 ساعة، أو نرسل بديلا إذا تعذّر التسليم.",
    },
    "refund": {
        "en": "Your refund request has been logged. Once the returned item is received and checked, the refund will be sent to your original payment method within 5-7 business days.",
        "ar": "تم تسجيل طلب الاسترجاع. بعد استلام المنتج المُرجَع وفحصه سيُعاد المبلغ إلى وسيلة الدفع الأصلية خلال 5-7 أيام عمل.",
    },
}


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        product=rng.choice(PRODUCTS),
        amount=rng.randint(20, 900),
        order=rng.randint(10000, 99999),
        code=rng.choice(["E101", "E204", "E500", "E731"]),
        days=rng.randint(3, 14),
    )


def build(n: int = 380, hard_share: float = 0.12, seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    rows, seen, attempts = [], set(), 0
    while len(rows) < n and attempts < n * 50:
        attempts += 1
        lang = rng.choice(["en", "ar"])
        priority = rng.choices(["high", "medium", "low"], weights=[0.2, 0.5, 0.3])[0]
        if rng.random() < hard_share:
            category, base = rng.choice([(c, t) for c, l, t in HARD if l == lang])
            intent, is_hard = "mixed", 1
        else:
            category, intent, en, ar = rng.choices(
                TEMPLATES, weights=[CATEGORY_WEIGHT[t[0]] for t in TEMPLATES]
            )[0]
            base, is_hard = _fill(en if lang == "en" else ar, rng), 0
        phrase = rng.choice(URGENCY[priority][lang])
        text = f"{phrase} {base}" if rng.random() < 0.4 else f"{base} {phrase}"
        text = " ".join(text.split())
        if text in seen:  # no duplicate texts -> avoids train/test leakage
            continue
        seen.add(text)
        rows.append(
            {
                "id": len(rows) + 1,
                "text": text,
                "language": lang,
                "category": category,
                "priority": priority,
                "intent": intent,
                "is_hard": is_hard,
                "resolution": RESOLUTIONS[category][lang],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df = build()
    path = DATA_DIR / "tickets.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} tickets to {path}")
    print(df["category"].value_counts().to_string())
    print(df["language"].value_counts().to_string())


if __name__ == "__main__":
    main()
