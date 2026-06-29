-- Migration 001 — multi-tenancy fix (Day-5 Phase-3)
--
-- The audited `transactions` table was (id, description, amount, date) with NO
-- owner column, so dashboard()/invest()/extract_bill() ran global queries and
-- every user saw and could delete every other user's transactions.
--
-- This migration adds a user_id foreign key scoping each transaction to its owner.
-- The application code (app.py, invest.py, extract_bill.py) now filters by
-- session['user_id'] on every read, insert, and delete.
--
-- Run once against the FinTrack MySQL database (e.g. `finase`):
--   mysql -u <user> -p <db> < db/migrations/001_add_user_id_to_transactions.sql

ALTER TABLE transactions
    ADD COLUMN user_id INT NULL AFTER id;

-- Backfill strategy: existing rows have no known owner. Assign them to a
-- reserved system user (id 0) or to a chosen account before enforcing NOT NULL.
-- UPDATE transactions SET user_id = 1 WHERE user_id IS NULL;

-- Once backfilled, enforce ownership + referential integrity:
-- ALTER TABLE transactions
--     MODIFY user_id INT NOT NULL,
--     ADD CONSTRAINT fk_transactions_user
--         FOREIGN KEY (user_id) REFERENCES users1(id) ON DELETE CASCADE;

-- Index the scoping column — every query now filters on it.
CREATE INDEX idx_transactions_user_id ON transactions (user_id);
