import { AttachImageIcon, CloseIcon, SettingsIcon } from '../../ui/icons';
import { useCallback, useEffect, useRef, useState, type RefObject, type SyntheticEvent } from 'react';
import { Button } from '../../ui/button';
import { SettingsPanel } from './settings-panel';
import { fmtNum } from '../format';
import { cn } from '../../ui/cn';
import type { Sampling } from '../types';

const COMPOSER_MAX_HEIGHT_PX = 200;

/** Exported so the smoke test can mount the real composer with one prop set
 *  rather than restating it and drifting from the component. */
export type ComposerProps = {
  sampling: Sampling;
  onSamplingChange: (next: Sampling) => void;
  onSubmit: (text: string, image: string | null) => void;
  streaming: boolean;
  onStop: () => void;
  vision: boolean;
  pendingImage: string | null;
  onImageFile: (file: File, label: string) => void;
  onRemoveImage: () => void;
  /** Reports a drop the composer cannot act on, so a file the reader let go of
   *  comes back with a reason instead of vanishing. */
  onDropRejected: (message: string) => void;
  onClearSystem: () => void;
  /** Tokens per second, or null when idle. */
  tps: number | null;
  settingsOpen: boolean;
  onToggleSettings: () => void;
  /** Bumped by the app to pull focus back to the composer after a turn. */
  focusToken: number;
};

/** The attached image, with the control that removes it. */
const ImagePreview = ({ src, onRemove }: { src: string; onRemove: () => void }) => (
  <div className="mx-auto block w-full agave-measure p-2 max-drawer:px-0">
    <div className="relative inline-block max-w-full">
      <img
        className="block max-h-20 max-w-50 rounded-lg border border-divider"
        src={src}
        alt="Attached image preview"
      />
      <Button
        type="button"
        size="iconRound"
        onClick={onRemove}
        aria-label="Remove image"
        title="Remove image"
        variant="destructiveSolid"
        className="absolute -end-2 -top-2"
      >
        <CloseIcon className="size-4" aria-hidden="true" />
      </Button>
    </div>
  </div>
);

/** The message field: Enter to send, Shift+Enter for a new line, paste or drop
 *  an image, and a height that follows its content. */
type PromptFieldProps = ComposerProps & {
  text: string;
  onText: (next: string) => void;
  area: RefObject<HTMLTextAreaElement | null>;
};

const PromptField = (props: PromptFieldProps) => (
  <textarea
    ref={props.area}
    id="msg"
    name="message"
    rows={1}
    value={props.text}
    dir="auto"
    placeholder="Prompt"
    aria-label="Message input"
    aria-describedby="input-hint"
    enterKeyHint="send"
    autoComplete="off"
    autoFocus
    disabled={props.streaming}
    onChange={function (event) { props.onText(event.target.value); }}
    onKeyDown={function (event) {
      // Ignore Enter while an IME composition is active (CJK input).
      // There, Enter confirms the conversion; it must not send.
      if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
        event.preventDefault();
        event.currentTarget.form?.requestSubmit();
      }
    }}
    onPaste={function (event) {
      /* Safari's DataTransferItemList is array-like without an iterator,
         so it is copied to a real array before the scan. */
      // oxlint-disable-next-line unicorn/no-useless-spread -- Safari has no iterator on this list
      for (const item of [...event.clipboardData.items]) {
        if (item.type.startsWith('image/')) {
          event.preventDefault();
          const pasted = item.getAsFile();
          if (pasted !== null) { props.onImageFile(pasted, 'Image pasted'); }
          return;
        }
      }
    }}
    // Min-width: 0 lets the field shrink below its intrinsic width.
    className="max-h-50 min-h-12 min-w-0 flex-1 resize-none rounded-lg border border-input bg-background px-4 py-3 leading-normal text-base text-foreground transition outline-none focus:border-primary focus:shadow-focus disabled:opacity-50 max-drawer:text-touch"
  />
);

/** The send key, which becomes the stop key while a turn is running. */
const SendControl = ({ streaming, canSend, onStop }: { streaming: boolean; canSend: boolean; onStop: () => void }) => (
  streaming ? (
    <Button type="button" variant="destructive" size="lg" onClick={onStop} title="Stop generation" aria-label="Stop generation">
      Stop
    </Button>
  ) : (
    <Button type="submit" variant="solid" size="lg" disabled={!canSend} title="Send message (Enter)" aria-label="Send message">
      Send
    </Button>
  )
);

/** The generation-speed readout, empty until a turn reports one.
 *
 *  Plain text, no live region. The number refreshes about once a second for
 *  the length of a turn, so a live region on it talks over the response
 *  announcement in `sr-announce` for as long as the turn runs. It also carried
 *  `role="status"` with an `aria-label`, which a live role cannot take as an
 *  accessible name, so the "Generation speed" label reached nobody. The rate
 *  is in the turn's own stats line (`/stats`) and in the response's stats
 *  block; the readout is the glanceable one, and the tooltip names it. */
const SpeedReadout = ({ tps }: { tps: number | null }) => (
  tps === null ? null : (
    <span title="Generation speed" className="flex items-center px-1 font-mono text-xs whitespace-nowrap text-primary">
      {`${fmtNum(tps, 1)} tok/s`}
    </span>
  )
);

/** The attach key, the settings key, and the send or stop control. */
const InputRow = (props: ComposerProps & {
  text: string;
  onText: (next: string) => void;
  canSend: boolean;
  area: RefObject<HTMLTextAreaElement | null>;
  fileInput: RefObject<HTMLInputElement | null>;
}) => (
  <div className="mx-auto flex w-full agave-measure items-end gap-2.5">
    <input
      ref={props.fileInput}
      type="file"
      accept="image/jpeg,image/png,image/gif,image/webp"
      /* Clipped rather than `display: none`. The attach key opens the picker
         through a ref, so the input was never reachable by keyboard, and a
         display-hidden input is not focusable at all, which is why there is no
         second tab stop. A clipped one keeps the programmatic path, exposes the
         control to assistive tech with its name, and is still not a stray stop
         in the tab order, because the attach key is the way in. */
      className="absolute size-px overflow-hidden border-0 p-0 opacity-0"
      tabIndex={-1}
      aria-hidden="true"
      onChange={function (event) {
        const file = event.target.files?.[0];
        /* Clear the field so picking the same image again is still a change
           event: an unsupported or oversized file is often re-picked smaller. */
        event.target.value = '';
        if (file !== undefined) { props.onImageFile(file, 'Image attached'); }
      }}
    />
    {props.vision ? (
      <Button
        type="button"
        size="icon"
        onClick={function () { props.fileInput.current?.click(); }}
        title="Attach image (or paste / drop)"
        aria-label="Attach image. You can also paste or drop an image."
        disabled={props.streaming}
      >
        <AttachImageIcon className="size-5" aria-hidden="true" />
      </Button>
    ) : null}
    <PromptField {...props} text={props.text} onText={props.onText} />
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
      <SettingsIcon className="size-5" aria-hidden="true" />
    </Button>
    <SendControl streaming={props.streaming} canSend={props.canSend} onStop={props.onStop} />
    <SpeedReadout tps={props.tps} />
  </div>
);

/** The form's own behavior: autosize, the drag-and-drop image path, and the
 *  focus pull the app requests after a turn. */
const useComposerForm = (props: ComposerProps, text: string, setText: (next: string) => void) => {
  const [dragOver, setDragOver] = useState(false);
  const area = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const focusedOnce = useRef(false);

  useEffect(() => {
    const element = area.current;
    if (!element) {return;}
    /* Re-measured when streaming flips too: the tok/s readout narrows the
       field mid-turn, and the wrapped hint text would keep that height. The box
       is border-box and scrollHeight excludes the border, so the border is
       added back, or the field is 2px short and scrolls on its first line. */
    element.style.height = 'auto';
    const border = element.offsetHeight - element.clientHeight;
    element.style.height = `${Math.min(element.scrollHeight + border, COMPOSER_MAX_HEIGHT_PX)}px`;
  }, [text, props.streaming]);

  useEffect(() => {
    if (!focusedOnce.current) {
      focusedOnce.current = true;
      return;
    }
    area.current?.focus();
  }, [props.focusToken]);

  const onSubmit = useCallback((event: SyntheticEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (props.streaming) {return;}
    if (!text.trim() && props.pendingImage === null) {return;}
    props.onSubmit(text.trim(), props.pendingImage);
    setText('');
  }, [props, setText, text]);

  return { dragOver, setDragOver, area, fileInput, onSubmit };
};

export const Composer = (props: ComposerProps) => {
  const [text, setText] = useState('');
  const form = useComposerForm(props, text, setText);
  const canSend = Boolean(text.trim() || props.pendingImage) && !props.streaming;

  return (
    <form
      aria-label="Chat"
      onSubmit={form.onSubmit}
      onDragOver={function (event) {
        event.preventDefault();
        if (props.vision) { form.setDragOver(true); }
      }}
      onDragLeave={function (event) {
        // Ignore leave events that stay inside the form (child to child flicker).
        if (event.relatedTarget instanceof Node && event.currentTarget.contains(event.relatedTarget)) {return;}
        form.setDragOver(false);
      }}
      onDrop={function (event) {
        /* Every drop is swallowed here, usable or not: the browser's default
           would navigate the tab away from the conversation. */
        event.preventDefault();
        form.setDragOver(false);
        const { files } = event.dataTransfer;
        if (files.length === 0) { return; }
        const [dropped] = files;
        if (!props.vision) {
          props.onDropRejected('This model cannot view images. Send the file as text instead.');
          return;
        }
        if (!dropped.type.startsWith('image/')) {
          props.onDropRejected('Only images can be attached here. JPEG, PNG, GIF, or WebP.');
          return;
        }
        props.onImageFile(dropped, 'Image dropped');
      }}
      className={cn(
        'relative z-10 border-t border-divider bg-card px-gutter pt-4 pb-5 transition-colors max-drawer:p-4',
        form.dragOver && 'border-primary bg-primary/10',
      )}
    >
      {props.pendingImage === null ? null : (
        <ImagePreview src={props.pendingImage} onRemove={props.onRemoveImage} />
      )}

      {props.settingsOpen ? (
        <SettingsPanel sampling={props.sampling} onChange={props.onSamplingChange} onClearSystem={props.onClearSystem} />
      ) : null}

      <InputRow
        {...props}
        text={text}
        onText={setText}
        canSend={canSend}
        area={form.area}
        fileInput={form.fileInput}
      />
      <p id="input-hint" className="mt-2 text-center font-mono text-2xs text-faint max-drawer:sr-only">
        Enter send &middot; Shift+Enter new line &middot; Esc stop &middot; /help commands
      </p>
    </form>
  );
};
