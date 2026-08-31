package com.dsrc.phone.config

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/**
 * Reading the link address from a pushed file.
 *
 * Before this, `SensingService` constructed `LinkConfig()` with defaults and the
 * default host is `127.0.0.1`, so nothing could point the app at a Jetson. The only
 * path that worked was `adb reverse`, which carries the data over USB -- the path
 * task 32 exists to avoid.
 */
class LinkConfigLoadTest {

    @get:Rule
    val folder = TemporaryFolder()

    private fun write(text: String): File {
        val dir = folder.newFolder()
        File(dir, LinkConfig.FILE_NAME).writeText(text)
        return dir
    }

    @Test
    fun `a pushed address is used, and says it came from the file`() {
        val dir = write("""{"host": "100.90.108.88", "port": 47811}""")
        val loaded = LinkConfig.load(dir)

        assertEquals("100.90.108.88", loaded.config.host)
        assertEquals(47811, loaded.config.port)
        assertEquals(LinkConfig.Source.FILE, loaded.source)
    }

    @Test
    fun `no file falls back to the defaults and says so`() {
        // The source travels with the value because a run that silently used loopback
        // and one that was pointed at a Jetson must not read alike in the record.
        val loaded = LinkConfig.load(folder.newFolder())

        assertEquals("127.0.0.1", loaded.config.host)
        assertEquals(LinkConfig.Source.DEFAULT, loaded.source)
    }

    @Test
    fun `a null directory falls back rather than throwing`() {
        // `getExternalFilesDir` returns null when external storage is unavailable.
        assertEquals(LinkConfig.Source.DEFAULT, LinkConfig.load(null).source)
    }

    @Test
    fun `the port may be omitted and takes the protocol default`() {
        val loaded = LinkConfig.load(write("""{"host": "100.90.108.88"}"""))
        assertEquals(LinkConfig.DEFAULT_PORT, loaded.config.port)
    }

    @Test
    fun `a malformed file is refused rather than defaulted`() {
        // A mistyped address that quietly became 127.0.0.1 would connect to nothing and
        // present as a link failure, which sends the search in the wrong direction.
        for (bad in listOf(
            "not json at all",
            """["host", "100.90.108.88"]""",
            """{"port": 47811}""",
            """{"host": 100}""",
            """{"host": "100.90.108.88", "port": "47811"}""",
        )) {
            val dir = write(bad)
            val thrown = assertThrows(
                "this should not have loaded: $bad",
                IllegalArgumentException::class.java,
            ) { LinkConfig.load(dir) }
            assertTrue(
                "the reason does not name the file or the fault: ${thrown.message}",
                (thrown.message ?: "").contains(LinkConfig.FILE_NAME),
            )
        }
    }

    @Test
    fun `the file goes through the same validation as a constructed instance`() {
        // A pushed file must not reach a state the constructor would refuse.
        for (bad in listOf(
            """{"host": "", "port": 47811}""",
            """{"host": "100.90.108.88", "port": 0}""",
            """{"host": "100.90.108.88", "port": 70000}""",
        )) {
            val dir = write(bad)
            assertThrows(
                "this should not have loaded: $bad",
                IllegalArgumentException::class.java,
            ) { LinkConfig.load(dir) }
        }
    }
}
