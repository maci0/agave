import { Image as ImageIcon, SlidersHorizontal, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { Button } from '../../ui/button';
import { SettingsPanel } from './settings-panel';
import { cn } from '../../ui/utils';
import type { Sampling } from '../types';

const COMPOSER_MAX_HEIGHT_PX = 200;
const IMAGE_PREVIEW_MAX_PX = 80;

type ComposerProps = {
  sampling: Sampling;
  onSamplingChange: (next: Sampling) => void;
  onSubmit: (text: string, image: string | null) => void;
  streaming: boolean;
  onStop: () => void;
  vision: boolean;
  pendingImage: string | null;
  onImageFile: (file: File, label: string) => void;
  onRemoveImage: () => void;
  onClearSystem: () => void;
  /** Tokens per second, or null when idle. */
  tps: number | null;
  settingsOpen: boolean;
  onToggleSettings: () => void;
  /** Bumped by the app to pull focus back to the composer after a turn. */
  focusToken: number;
};

export const Composer = (props: ComposerProps) => {
  const [text, setText] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const area = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const focusedOnce = useRef(false);

  const canSend = Boolean(text.trim() || props.pendingImage) && !props.streaming;

  useEffect(function () {
    const element = area.current;
    if (!element) {return;}
    element.style.height = 'auto';
    element.style.height = `${Math.min(element.scrollHeight, COMPOSER_MAX_HEIGHT_PX)}px`;
  }, [text]);

  useEffect(function () {
    if (!focusedOnce.current) {
      focusedOnce.current = true;
      return;
    }
    area.current?.focus();
  }, [props.focusToken]);

  return (
    <form
      aria-label="Chat"
      onSubmit={function (event) {
        event.preventDefault();
        if (!canSend) {return;}
        props.onSubmit(text.trim(), props.pendingImage);
        setText('');
      }}
      onDragOver={function (event) {
        event.preventDefault();
        if (props.vision) { setDragOver(true); }
      }}
      onDragLeave={function (event) {
        // Ignore leave events that stay inside the form (child to child flicker).
        if (event.relatedTarget instanceof Node && event.currentTarget.contains(event.relatedTarget)) {return;}
        setDragOver(false);
      }}
      onDrop={function (event) {
        event.preventDefault();
        setDragOver(false);
        const file = event.dataTransfer?.files?.[0];
        if (file?.type.startsWith('image/')) { props.onImageFile(file, 'Image dropped'); }
      }}
      className={cn(
        'relative z-10 border-t border-border bg-card px-6 pt-4 pb-5 transition-colors max-drawer:p-4',
        dragOver && 'border-primary bg-primary/10',
      )}
    >
      {props.pendingImage ? (
        <div className="mx-auto block w-full max-w-prose p-2 max-drawer:px-0">
          <div className="relative inline-block max-w-full">
            <img
              className="block max-w-[200px] rounded-lg border border-border"
              style={{ maxHeight: `${IMAGE_PREVIEW_MAX_PX}px` }}
              src={props.pendingImage}
              alt="Attached image preview"
            />
            <Button
              type="button"
              size="iconSm"
              onClick={props.onRemoveImage}
              aria-label="Remove image"
              title="Remove image"
              // The glyph uses --background on --destructive (5.98:1) and on the
              // lighter hover red (8.47:1): white measured only 2.99:1, below the
              // 1.4.3 minimum for this label.
              className="absolute -end-2 -top-2 rounded-pill border-none bg-destructive text-background hover:bg-destructive-foreground"
            >
              <X className="size-4" aria-hidden="true" />
            </Button>
          </div>
        </div>
      ) : null}

      {props.settingsOpen ? (
        <SettingsPanel sampling={props.sampling} onChange={props.onSamplingChange} onClearSystem={props.onClearSystem} />
      ) : null}

      <div className="mx-auto flex w-full max-w-prose items-end gap-2.5">
        <input
          ref={fileInput}
          type="file"
          accept="image/jpeg,image/png,image/gif,image/webp"
          className="hidden"
          aria-label="Attach image"
          onChange={function (event) {
            const file = event.target.files?.[0];
            if (file) { props.onImageFile(file, 'Image attached'); }
          }}
        />
        {props.vision ? (
          <Button
            type="button"
            size="icon"
            onClick={function () { fileInput.current?.click(); }}
            title="Attach image (or paste / drop)"
            aria-label="Attach image. You can also paste or drop an image."
            disabled={props.streaming}
          >
            <ImageIcon className="size-5" aria-hidden="true" />
          </Button>
        ) : null}
        <textarea
          ref={area}
          id="msg"
          name="message"
          rows={1}
          value={text}
          dir="auto"
          placeholder="Prompt"
          aria-label="Message input"
          aria-describedby="input-hint"
          enterKeyHint="send"
          autoComplete="off"
          autoFocus
          disabled={props.streaming}
          onChange={function (event) { setText(event.target.value); }}
          onKeyDown={function (event) {
            // Ignore Enter while an IME composition is active (CJK input): there
            // it confirms the conversion, it must not send the message.
            if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}
          onPaste={function (event) {
            const items = event.clipboardData?.items;
            if (!items) {return;}
            for (let index = 0; index < items.length; index++) {
              const item = items[index];
              if (item?.type.startsWith('image/')) {
                event.preventDefault();
                const file = item.getAsFile();
                if (file) { props.onImageFile(file, 'Image pasted'); }
                return;
              }
            }
          }}
          // min-width: 0 lets the field shrink below its intrinsic min-content
          // width, keeping the row inside a 320px viewport (WCAG 1.4.10).
          className="max-h-50 min-h-12 min-w-0 flex-1 resize-none rounded-lg border border-input bg-background px-4 py-3 leading-relaxed text-base text-foreground transition-[border-color,box-shadow] outline-none placeholder:text-faint focus:border-primary focus:shadow-[0_0_0_3px_color-mix(in_oklab,var(--color-primary)_10%,transparent)] disabled:opacity-50 max-drawer:text-[16px]"
        />
        <Button
          type="button"
          size="icon"
          active={props.settingsOpen}
          onClick={props.onToggleSettings}
          title="Sampling settings"
          aria-label={props.settingsOpen ? 'Close sampling settings' : 'Open sampling settings'}
          aria-expanded={props.settingsOpen}
          aria-controls="settings-panel"
        >
          <SlidersHorizontal className="size-5" aria-hidden="true" />
        </Button>
        {props.streaming ? (
          <Button type="button" variant="destructive" size="lg" onClick={props.onStop} title="Stop generation" aria-label="Stop generation">
            Stop
          </Button>
        ) : (
          <Button type="submit" variant="primaryOutline" size="lg" disabled={!canSend} title="Send message (Enter)" aria-label="Send message">
            Send
          </Button>
        )}
        {props.tps === null ? null : (
          <span role="status" aria-label="Generation speed" className="flex items-center px-1 font-mono text-xs whitespace-nowrap text-primary">
            {`${props.tps.toFixed(1)} tok/s`}
          </span>
        )}
      </div>
      <p id="input-hint" className="mx-auto mt-2 w-full max-w-prose text-center font-mono text-2xs text-faint max-drawer:sr-only">
        Enter to send &middot; Shift+Enter for new line &middot; Escape to stop &middot; /help for commands
      </p>
    </form>
  );
};
