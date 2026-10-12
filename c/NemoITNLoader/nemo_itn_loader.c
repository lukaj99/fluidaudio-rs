#include "nemo_itn_loader.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define NEMO_ITN_DYLIB "libfluidaudio_nemo_itn.dylib"

static void *nemo_handle;
static pthread_once_t nemo_once = PTHREAD_ONCE_INIT;

static void nemo_load(void) {
    const char *override = getenv("FLUIDAUDIO_NEMO_ITN_DYLIB");
    if (override != NULL && override[0] != '\0') {
        nemo_handle = dlopen(override, RTLD_NOW | RTLD_LOCAL);
        return;
    }
    static const char *const candidates[] = {
        "@executable_path/../Frameworks/" NEMO_ITN_DYLIB,
        "@executable_path/" NEMO_ITN_DYLIB,
        "@executable_path/../" NEMO_ITN_DYLIB,
    };
    for (size_t i = 0; i < sizeof candidates / sizeof candidates[0] && nemo_handle == NULL; i++) {
        nemo_handle = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
    }
}

int fluidaudio_nemo_itn_available(void) {
    pthread_once(&nemo_once, nemo_load);
    return nemo_handle != NULL;
}

static void *nemo_symbol(const char *name) {
    return fluidaudio_nemo_itn_available() ? dlsym(nemo_handle, name) : NULL;
}

// Forwarders for the C API in text-processing-rs's nemo_text_processing.h.
// Each looks its target up per call: dlsym is cheap next to normalization.

#define NEMO_FORWARD(ret, fallback, name, params, args)                  \
    ret name params {                                                    \
        ret(*target) params = (ret(*) params)nemo_symbol(#name);         \
        return target != NULL ? target args : fallback;                  \
    }

#define NEMO_FORWARD_VOID(name, params, args)                            \
    void name params {                                                   \
        void(*target) params = (void(*) params)nemo_symbol(#name);       \
        if (target != NULL) target args;                                 \
    }

NEMO_FORWARD(char *, NULL, nemo_normalize, (const char *input), (input))
NEMO_FORWARD(char *, NULL, nemo_normalize_sentence, (const char *input), (input))
NEMO_FORWARD(char *, NULL, nemo_normalize_with_options,
             (const char *input, uint32_t concat_compound_numbers, uint32_t disable_bare_second),
             (input, concat_compound_numbers, disable_bare_second))
NEMO_FORWARD(char *, NULL, nemo_normalize_sentence_with_options,
             (const char *input, uint32_t concat_compound_numbers, uint32_t max_span_tokens,
              uint32_t disable_bare_second),
             (input, concat_compound_numbers, max_span_tokens, disable_bare_second))
NEMO_FORWARD(char *, NULL, nemo_normalize_sentence_lang, (const char *input, const char *lang),
             (input, lang))
NEMO_FORWARD_VOID(nemo_add_rule, (const char *spoken, const char *written), (spoken, written))
NEMO_FORWARD(int32_t, 0, nemo_remove_rule, (const char *spoken), (spoken))
NEMO_FORWARD_VOID(nemo_clear_rules, (void), ())
NEMO_FORWARD(uint32_t, 0, nemo_rule_count, (void), ())
NEMO_FORWARD(char *, NULL, nemo_tn_normalize, (const char *input), (input))
NEMO_FORWARD(char *, NULL, nemo_tn_normalize_sentence, (const char *input), (input))
NEMO_FORWARD(char *, NULL, nemo_tn_normalize_sentence_with_max_span,
             (const char *input, uint32_t max_span_tokens), (input, max_span_tokens))
NEMO_FORWARD(char *, NULL, nemo_tn_normalize_lang, (const char *input, const char *lang),
             (input, lang))
NEMO_FORWARD(char *, NULL, nemo_tn_normalize_sentence_lang, (const char *input, const char *lang),
             (input, lang))
NEMO_FORWARD(char *, NULL, nemo_tn_normalize_sentence_with_max_span_lang,
             (const char *input, const char *lang, uint32_t max_span_tokens),
             (input, lang, max_span_tokens))
NEMO_FORWARD(char *, NULL, nemo_tn_fst, (const char *input, const char *lang), (input, lang))
NEMO_FORWARD_VOID(nemo_free_string, (char *s), (s))
NEMO_FORWARD(const char *, NULL, nemo_version, (void), ())
