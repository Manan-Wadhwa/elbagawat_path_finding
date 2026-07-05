# Phase 0: Step 0e — Inspect Excel Database and Schema Validation

## Metadata
- **Date**: 2026-06-24
- **Database File**: [2026 El Bagawat Database Draft 1.xlsx](file:///C:/Users/Public/LAMP_DataStore/ElBagawat/100_Data/120_SiteReport/2026%20El%20Bagawat%20Database%20Draft%201.xlsx) (Updated based on user/assistant feedback)

---

## 1. Sheet Structure and Verification of Assistant's Suggestion
The assistant suggested that the correct Excel file is `2026 El Bagawat Database Draft 1.xlsx` instead of `Bagawat Data From Excavation Report.xlsx`. 

After examining the file, this suggestion is **100% correct**. The file contains the correct sheet structure and row count assumed in the master technical overview plan:
- **Actual Sheets**:
  - `Database Full` (342 rows, 42 columns) — The main attribute database.
  - `Building Assignments` (39 rows, 4 columns)
  - `Sheet4` (13 rows, 6 columns)

---

## 2. Schema and Column Names
The main sheet is `Database Full` and has the following key columns for our spatial joining and modeling:
- **ID Column**: `Chapel Number (according to Fakhry)` (contains chapel IDs 1 to 342, with 341 non-nulls)
- **Entrance Direction Column**: **`Entrace Direction`** (note the exact spelling: *Entrace* without the second 'n').
- **Architectural Type Column**: `Type` (232 non-nulls)

---

## 3. Data Distribution of Entrance Direction
Out of 342 records, there are **215 non-null entrance directions**. Cleaning the string values (handling trailing spaces and lowercasing) yields the following distribution:
- **South**: 89 records (84 exact `South`, 5 `South ` with trailing spaces)
- **East**: 61 records (60 exact `East`, 1 `East (table said south?) pg. 120`)
- **West**: 63 records (59 exact `West`, 1 `west` lowercase, 1 `West (table says east) pg. 121`, 1 `West (Fig. 120)`)
- **South & West**: 2 records
- **East/West Compound (Secondary)**: 1 record (`East, secondary west entrance - court entrance (off the court)`)
- **Missing / Unrecorded (NaN)**: 127 records

This matches the expected distribution in the technical plan overview (South ~84-91, East ~60-62, West ~59-62).

---

## 4. Viability of Phase 3c (Attribute-Driven Fallback)
This audit confirms that the **Attribute-Driven Fallback** is highly viable:
- We can derive wall offsets for **215 chapels** if they are skipped by the manual annotator or lack visible marks on the PDF scan.
- For the remaining **127 chapels** (where `Entrace Direction` is `NaN`), we must fall back to the **geometric centroid** of the footprint polygon as the last-resort entrance coordinate (confidence = 0.30).
