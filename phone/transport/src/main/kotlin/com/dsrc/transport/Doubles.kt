package com.dsrc.transport

/**
 * Formats a double exactly as CPython's `json.dumps` does.
 *
 * The header is canonical JSON, so a float has one correct spelling and the two
 * implementations must agree on it byte for byte. They do not by default:
 * `Double.toString()` emits `1.5E-5` where Python emits `1.5e-05`, and `1.0E7` where
 * Python emits `10000000.0`.
 *
 * What makes that dangerous rather than merely annoying is *where* they agree. Every
 * float in `specs/transport_golden_frames.json` falls in the range where the two
 * spellings coincide, so the vectors pass while the formatter is wrong, and the first
 * divergence would appear on real data — a near-zero IMU gyro reading sits squarely in
 * the range where Kotlin switches to exponent notation and Python does not.
 *
 * Python's rule, in two parts:
 *
 *  - the digits are the shortest decimal that round-trips to the same double, computed
 *    here rather than taken from `Double.toString()`. That shortcut is tempting and
 *    wrong: on JDK 17 `toString` is documented only to distinguish the value from its
 *    neighbours, not to be minimal, and it demonstrably is not. It spells
 *    `Double.MIN_VALUE` as `4.9E-324` where the shortest form is `5e-324`, and emits 17
 *    significant digits for values needing 15. Five of 1,256 reference cases diverged
 *    on exactly this. (JDK 19 fixed the implementation; 17 is what is installed.)
 *  - the *layout* is fixed-point when the value's decimal exponent is in `[-4, 16)` and
 *    scientific otherwise, with the exponent signed and at least two digits.
 *
 * `1e-4` prints as `0.0001` and `1e-5` as `1e-05`; `1e15` prints as
 * `1000000000000000.0` and `1e16` as `1e+16`. Those four are the boundaries.
 */
object Doubles {

    /** Below this decimal exponent Python switches to scientific notation. */
    private const val MIN_FIXED_EXPONENT = -4

    /** At or above this decimal exponent Python switches to scientific notation. */
    private const val MAX_FIXED_EXPONENT = 16

    /** Seventeen significant digits always round-trip a double. */
    private const val MAX_SIGNIFICANT_DIGITS = 17

    /**
     * The canonical spelling of [value].
     *
     * @throws IllegalArgumentException for NaN or an infinity, which must never reach
     *   the wire: Python writes a bare `NaN` token that a strict parser elsewhere
     *   rejects, so an encoder converts a non-finite value to `null` *before* framing.
     */
    fun format(value: Double): String {
        require(!value.isNaN()) { "NaN must not reach the wire" }
        require(!value.isInfinite()) { "$value must not reach the wire" }

        // -0.0 has to be handled here: it is not caught by `== 0.0`, and Python does
        // print the sign.
        if (value == 0.0) return if (1.0 / value < 0) "-0.0" else "0.0"

        val negative = value < 0
        val decimal = Decimal.of(kotlin.math.abs(value))
        val body = if (decimal.exponent >= MIN_FIXED_EXPONENT && decimal.exponent < MAX_FIXED_EXPONENT) {
            decimal.fixed()
        } else {
            decimal.scientific()
        }
        return if (negative) "-$body" else body
    }

    /**
     * Shortest round-trip digits, with the decimal exponent of the leading digit.
     *
     * `digits` carries no sign, no point and no leading zero; `exponent` is `e` in
     * `0.d1d2... x 10^(e+1)`, i.e. `123` with exponent `2` means `123.0`.
     */
    private class Decimal(val digits: String, val exponent: Int) {

        fun fixed(): String {
            if (exponent >= 0) {
                val intLength = exponent + 1
                return if (digits.length <= intLength) {
                    // Pad out to the point, then Python's mandatory ".0".
                    digits + "0".repeat(intLength - digits.length) + ".0"
                } else {
                    digits.substring(0, intLength) + "." + digits.substring(intLength)
                }
            }
            // exponent is -1..-4 here, so there are leading zeros after the point.
            return "0." + "0".repeat(-exponent - 1) + digits
        }

        fun scientific(): String {
            val mantissa = if (digits.length == 1) digits else digits[0] + "." + digits.substring(1)
            val sign = if (exponent < 0) "-" else "+"
            val magnitude = kotlin.math.abs(exponent).toString().padStart(2, '0')
            return "${mantissa}e$sign$magnitude"
        }

        companion object {
            /**
             * The shortest decimal that parses back to exactly [magnitude].
             *
             * Found by rounding the value's exact binary expansion to 1, 2, 3 ...
             * significant digits and taking the first that round-trips. Seventeen always
             * suffices for a double, so the loop is bounded and the last iteration is
             * exact by construction.
             *
             * Cost is a handful of BigDecimal roundings per float. At the rates on this
             * wire -- six floats per IMU sample at 50 Hz is the worst case -- that is
             * not close to mattering, and correctness here is not negotiable: the two
             * implementations must agree byte for byte or the frame hashes differ.
             */
            fun of(magnitude: Double): Decimal {
                val exact = java.math.BigDecimal(magnitude)
                var shortest: java.math.BigDecimal = exact
                for (precision in 1..MAX_SIGNIFICANT_DIGITS) {
                    // HALF_EVEN, not MathContext(int)'s default of HALF_UP. Where two
                    // equally short decimals both round-trip, Python takes the one
                    // nearer the exact binary value, and HALF_UP breaks that tie the
                    // other way: -1054347931188540.25 spells as ...40.3 under HALF_UP
                    // and ...40.2 under HALF_EVEN, which is what Python emits.
                    val candidate = exact.round(
                        java.math.MathContext(precision, java.math.RoundingMode.HALF_EVEN)
                    )
                    if (candidate.toDouble() == magnitude) {
                        shortest = candidate
                        break
                    }
                }

                // stripTrailingZeros so 35.0 and 3.5e1 both reduce to digits "35",
                // leaving the layout entirely to the renderer.
                val stripped = shortest.stripTrailingZeros()
                val digits = stripped.unscaledValue().abs().toString()
                // precision - scale - 1 is the decimal exponent of the leading digit.
                val exponent = stripped.precision() - stripped.scale() - 1
                return Decimal(digits, exponent)
            }
        }
    }
}
