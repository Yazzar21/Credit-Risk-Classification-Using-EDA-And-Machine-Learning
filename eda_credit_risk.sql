-- 2. Menghitung jumlah dan rasio status pinjaman (Lancar vs Default)
SELECT 
-- 3. Analisis Tingkat Default dan Rata-rata Pinjaman Berdasarkan Tujuan Pinjaman
SELECT 
    loan_intent AS tujuan_pinjaman,
    COUNT(*) AS total_nasabah,
    SUM(loan_status) AS total_gagal_bayar,
    ROUND(AVG(loan_status) * 100.0, 2) AS default_rate_persen,
    ROUND(AVG(loan_amnt), 2) AS rata_rata_nominal_pinjaman
FROM credit_risk_bersih
GROUP BY loan_intent
ORDER BY default_rate_persen DESC;

    loan_status,
    COUNT(*) AS total_nasabah,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM credit_risk_bersih), 2) AS persentase
FROM credit_risk_bersih
GROUP BY loan_status;

-- 4. Analisis Tingkat Risiko Berdasarkan Peringkat Kredit (Loan Grade)
SELECT 
    loan_grade AS peringkat_kredit,
    COUNT(*) AS total_nasabah,
    SUM(loan_status) AS total_gagal_bayar,
    ROUND(AVG(loan_status) * 100.0, 2) AS default_rate_persen,
    ROUND(AVG(loan_int_rate), 2) AS rata_rata_suku_bunga
FROM credit_risk_bersih
GROUP BY loan_grade
ORDER BY loan_grade ASC;